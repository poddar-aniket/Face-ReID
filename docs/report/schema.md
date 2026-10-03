# Data model (SQLite, `data/face_index.db`)

images (1) ──< faces (N)

| images     |                              |
|------------|------------------------------|
| image_id   | TEXT PK (filename stem)      |
| image_path | TEXT UNIQUE (repo-relative)  |

| faces      |                                          |
|------------|------------------------------------------|
| face_id    | TEXT PK (e.g. img014_face2)              |
| image_id   | TEXT FK -> images.image_id, ON DELETE CASCADE |
| embedding  | BLOB (512 x float32, L2-normalized)      |
| x1,y1,x2,y2| INTEGER (pixel bbox in original image)  |
| det_score  | REAL (detection confidence)              |

Search: load all embeddings into an (N, 512) matrix, scores = matrix @ query,
filter by threshold, keep the best face per image, sort desc, take top_k.