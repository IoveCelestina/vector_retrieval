#pragma once
#include<cstdint>
#include<string>
#include<utility>
#include<vector>

namespace vecsearch {
	using Id = std::uint32_t;

	//目前只支持L2
	enum class Metric :std::uint8_t {//enum class比enum更安全枚举，一定得写Metric::L2不会有别的枚举名字冲突
		L2 = 0,//:2选项的值是0
	};

	//单条搜索结果:返回(id,dist),内存对其的问题?
	struct Neighbor {
		Id id;
		float dist;//距离平方，不用开方更快

	};

	//Index配置,dim:向量唯独，目前128,metric:距离类型,normalize:归一化,cosine常用
	struct IndexConfig {
		int dim = 0;
		Metric metric = Metric::L2;
		bool normalize = false;
	};

	//Index的统一接口(所有索引都要实现)
	class  IIndex {
	public:
		virtual  ~IIndex() = default;
		//读配置(维度,距离类型)
		virtual IndexConfig config() const = 0;//虚函数，=0表示纯虚函数，这种抽象类没有实体，const函数只读，不修改内部状态

		//当前已经插入向量数量
		virtual std::size_t size() const = 0;

		//清空索引
		virtual void clear() = 0;


		//批量加入向量,ids长度为n，vectors:长度 = n*dim,平铺，第i调向量的起始地址 = &vectors[i*dim]
		//约束:ids.size(*dim == vectors.size(),dim必须与config().dim一致,ids要求唯一，不要求连续

		virtual void add_batch(const std::vector<Id>& ids,const std::vector<float>&vectors) = 0;

		//单条query搜索topk
		//q:指向dim个float的数组,topk返回K个最相邻
		//返回vector<Neighbor>,按dist升序排列
		virtual std::vector<Neighbor> search_one(const float *q,int topk) const = 0;

		//多条query,默认实现可以循环调用search_one
		virtual std::vector<std::vector<Neighbor>> search_batch(const float *queries,int num_queries,int topk) const = 0;


		//用于打印记录到csv
		virtual std::string name() const = 0;

		//打印参数
		virtual std::string params() const = 0;
	};



}