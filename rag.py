import os
import hashlib
import math
from typing import List, Dict, Any, Optional

# Attempt to import Qdrant, fallback if not available
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

class FallbackVectorStore:
    """A pure Python in-memory vector store that mimics Qdrant client behaviour."""
    def __init__(self):
        self.collections: Dict[str, List[Dict[str, Any]]] = {}

    def recreate_collection(self, collection_name: str, vectors_config: Any) -> bool:
        self.collections[collection_name] = []
        return True

    def upsert(self, collection_name: str, points: List[Any]) -> Any:
        if collection_name not in self.collections:
            self.collections[collection_name] = []
        for point in points:
            # We assume point is a PointStruct or similar dictionary-like object
            if hasattr(point, 'id'):
                point_dict = {
                    'id': point.id,
                    'vector': point.vector,
                    'payload': point.payload
                }
            else:
                point_dict = point
            self.collections[collection_name].append(point_dict)
        return len(points)

    def search(self, collection_name: str, query_vector: List[float], limit: int = 3) -> List[Any]:
        if collection_name not in self.collections:
            return []
        
        # Calculate cosine similarity
        results = []
        for point in self.collections[collection_name]:
            vec = point['vector']
            # Dot product
            dot = sum(a*b for a, b in zip(query_vector, vec))
            norm_a = math.sqrt(sum(a*a for a in query_vector))
            norm_b = math.sqrt(sum(b*b for b in vec))
            
            similarity = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
            
            # Mimic Qdrant ScoredPoint
            class ScoredPoint:
                def __init__(self, id, payload, score):
                    self.id = id
                    self.payload = payload
                    self.score = score
            
            results.append(ScoredPoint(point['id'], point['payload'], similarity))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

class SimpleRAG:
    """Manages local vector store using Qdrant (or fallback) seeded with style guides and Remotion snippets."""
    
    def __init__(self, use_gemini_embeddings: bool = False):
        self.dimension = 768 # Standard Gemini embedding dimension
        self.use_gemini_embeddings = use_gemini_embeddings
        
        if QDRANT_AVAILABLE:
            try:
                self.client = QdrantClient(":memory:")
                self.is_fallback = False
            except Exception:
                self.client = FallbackVectorStore()
                self.is_fallback = True
        else:
            self.client = FallbackVectorStore()
            self.is_fallback = True

        self._initialize_collections()
        self.seed_database()

    def _initialize_collections(self):
        if self.is_fallback:
            self.client.recreate_collection("style_guides", None)
            self.client.recreate_collection("remotion_api", None)
        else:
            from qdrant_client.models import Distance, VectorParams
            self.client.recreate_collection(
                collection_name="style_guides",
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
            )
            self.client.recreate_collection(
                collection_name="remotion_api",
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
            )

    def get_embedding(self, text: str) -> List[float]:
        """Generates embedding vector. Uses Gemini API if available, else falls back to hash-based TF embedding."""
        if self.use_gemini_embeddings and os.getenv("GEMINI_API_KEY"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text
                )
                embedding = response.get('embedding', [])
                if len(embedding) == self.dimension:
                    return embedding
            except Exception:
                pass # Fallback to local embedding
                
        # Pure Python Local Term Frequency-based embedding
        words = text.lower().split()
        vector = [0.0] * self.dimension
        if not words:
            return vector
        for word in words:
            # Deterministic hash function to map word to index
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            index = h % self.dimension
            vector[index] += 1.0
            
        # Normalize
        norm = math.sqrt(sum(x*x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector

    def seed_database(self):
        """Seeds style guides and Remotion API documentation snippets."""
        # 1. Style Guides
        style_guides = {
            "cinematic": (
                "Cinematic Video Style: Slow pacing, 4.0 to 6.0 seconds per slide. "
                "Elegant, warm tones and rich visuals. Use 'fade' (crossfade) transitions with a "
                "duration of 1.0 to 1.5 seconds. Captions should be minimal, emotional, and centered "
                "at the bottom of the screen. Use elegant serif fonts such as 'Playfair Display' "
                "or 'Georgia'. Keep font colors soft, like warm white '#fdfbf7' or light cream."
            ),
            "upbeat": (
                "Upbeat Video Style: Fast pacing, 1.0 to 2.0 seconds per slide. "
                "Saturated colors, high contrast, and energetic visuals. Use 'zoom' or 'slide' transitions "
                "with a very quick duration of 0.2 to 0.4 seconds, or direct cuts ('none'). Captions should "
                "be bold, loud, punchy, and center-aligned or top-aligned. Use bold sans-serif fonts such as "
                "'Impact' or 'Montserrat'. Use vibrant font colors like bright yellow '#ffcc00' or hot pink '#ff007f'."
            ),
            "corporate": (
                "Corporate Video Style: Medium pacing, 3.0 to 4.0 seconds per slide. "
                "Clean, professional, and balanced visuals with cool or neutral tones. Use subtle 'slide' "
                "or 'fade' transitions of 0.8 seconds duration. Captions should be structured, informative, "
                "and professional, aligned bottom-left or bottom-center. Use clean modern sans-serif fonts "
                "like 'Inter' or 'Roboto'. Use standard white '#ffffff' or professional light blue '#e0f2fe' "
                "with subtle text shadows."
            )
        }

        # 2. Remotion API reference snippets
        remotion_api = {
            "composition": (
                "Remotion Composition API: The <Composition> component registers a video composition in src/Root.tsx. "
                "It takes props: id (unique string), component (React component), durationInFrames (number of frames, "
                "calculated as duration_in_sec * fps), fps (usually 30), width (pixels, e.g. 1920 or 1080), and height "
                "(pixels, e.g. 1080 or 1920). Example:\n"
                "```tsx\n"
                "import { Composition } from 'remotion';\n"
                "import { Slideshow } from './Slideshow';\n"
                "export const Root = () => {\n"
                "  return (\n"
                "    <Composition\n"
                "      id=\"Slideshow\"\n"
                "      component={Slideshow}\n"
                "      durationInFrames={300}\n"
                "      fps={30}\n"
                "      width={1080}\n"
                "      height={1920}\n"
                "    />\n"
                "  );\n"
                "};\n"
                "```"
            ),
            "sequence": (
                "Remotion Sequence API: The <Sequence> component is used to layer and sequence components in time. "
                "It takes props: from (frame number where the sequence starts, 0-indexed) and durationInFrames (how long "
                "the sequence lasts). It creates a new local time coordinate system starting at frame 0 for its children. Example:\n"
                "```tsx\n"
                "import { Sequence } from 'remotion';\n"
                "export const Slideshow = () => {\n"
                "  return (\n"
                "    <>\n"
                "      <Sequence from={0} durationInFrames={90}>\n"
                "        <Slide1 />\n"
                "      </Sequence>\n"
                "      <Sequence from={90} durationInFrames={90}>\n"
                "        <Slide2 />\n"
                "      </Sequence>\n"
                "    </>\n"
                "  );\n"
                "};\n"
                "```"
            ),
            "interpolate": (
                "Remotion Interpolation API: The interpolate function maps an input value (usually the current frame) "
                "from an input range to an output range. It takes: value (number), inputRange (array of numbers), "
                "outputRange (array of numbers), and an optional options object (e.g. { extrapolateLeft: 'clamp', "
                "extrapolateRight: 'clamp' }). Example for opacity fade:\n"
                "```tsx\n"
                "import { interpolate, useCurrentFrame } from 'remotion';\n"
                "export const Slide = () => {\n"
                "  const frame = useCurrentFrame();\n"
                "  const opacity = interpolate(frame, [0, 15], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' });\n"
                "  return <div style={{ opacity }}>Content</div>;\n"
                "};\n"
                "```"
            ),
            "image": (
                "Remotion Image API: Instead of standard HTML <img> elements, use the <Img> component from 'remotion'. "
                "This ensures proper asset resolution and loading during rendering. It takes standard img attributes. Example:\n"
                "```tsx\n"
                "import { Img, staticFile } from 'remotion';\n"
                "export const Photo = ({ path }) => {\n"
                "  return <Img src={staticFile(path)} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />;\n"
                "};\n"
                "```"
            ),
            "spring": (
                "Remotion Spring API: The spring function creates a smooth spring physics-based animation value based on "
                "the current frame and FPS. It accepts an object configuration: { frame, fps, config: { damping: 10 } }. "
                "It returns a value from 0 to 1. Great for popping captions. Example:\n"
                "```tsx\n"
                "import { spring, useCurrentFrame, useVideoConfig } from 'remotion';\n"
                "export const Caption = ({ text }) => {\n"
                "  const frame = useCurrentFrame();\n"
                "  const { fps } = useVideoConfig();\n"
                "  const scale = spring({ frame, fps, config: { damping: 12 } });\n"
                "  return <div style={{ transform: `scale(${scale})` }}>{text}</div>;\n"
                "};\n"
                "```"
            ),
            "use_current_frame": (
                "Remotion useCurrentFrame API: The useCurrentFrame() hook returns the current frame of the rendering context, "
                "which is an integer starting from 0 and incrementing by 1 for each frame of the composition. Use this "
                "hook inside visual components to drive transitions or text animations. Example:\n"
                "```tsx\n"
                "import { useCurrentFrame } from 'remotion';\n"
                "export const MyComponent = () => {\n"
                "  const frame = useCurrentFrame();\n"
                "  return <div>Current frame is {frame}</div>;\n"
                "};\n"
                "```"
            )
        }

        # Seed style guides
        style_points = []
        for i, (name, content) in enumerate(style_guides.items()):
            vec = self.get_embedding(content)
            payload = {"style_name": name, "content": content}
            if self.is_fallback:
                style_points.append({"id": i, "vector": vec, "payload": payload})
            else:
                style_points.append(PointStruct(id=i, vector=vec, payload=payload))
        self.client.upsert(collection_name="style_guides", points=style_points)

        # Seed Remotion API reference
        api_points = []
        for i, (name, content) in enumerate(remotion_api.items()):
            vec = self.get_embedding(content)
            payload = {"api_name": name, "content": content}
            if self.is_fallback:
                api_points.append({"id": i, "vector": vec, "payload": payload})
            else:
                api_points.append(PointStruct(id=i, vector=vec, payload=payload))
        self.client.upsert(collection_name="remotion_api", points=api_points)

    def retrieve(self, collection_name: str, query: str, limit: int = 1) -> List[Dict[str, Any]]:
        """Retrieves documents from a specific collection based on semantic similarity of the query."""
        query_vector = self.get_embedding(query)
        hits = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return [hit.payload for hit in hits]
