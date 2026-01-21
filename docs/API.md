- Version: v0.1
- Metric: L2(squared) 欧几里得距离
- Dim: fixed at creation
- Id: uint32_t
- Add
    - add(id,vec)
    - add_batch(ids,vectors row-major) 一次性添加多条,ids长度为n的索引数组
      - row-major:按行平铺,假设
      - v0=[1,2,3,4],v1=[5,6,7,8],v2 = [9,10,11,12]
      - 展开后变为 vectors = [1,2,3,4,5,6,7,8,9,10,11,12]
    - duplicate id:error
  
- Search
  - search(query,topk) neighors sorted asc by distance
  - search_batch(queries row-major,nq,topk)
  - topk>size: redturn min(topk,size)

- HNSW Options
  - build: M,efConstruction
  - Search








