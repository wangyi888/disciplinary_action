package disciplinary_action;
/**
 * 违纪行为识别
 * author:wangyi
 */
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import io.github.yizhiru.thulac4j.Segmenter;

public class DisciplinaryModel {
	 public static void main(String[] args)throws Exception {
		 File directory = new File("");//参数为空
		 String basePath =directory.getAbsolutePath();//绝对路径;
		 String resourcePath = basePath+"/src/main/resources/";
		 //实例化预测网络类0321
		NeuralNetworkModel neuralNetworkModel = new NeuralNetworkModel(
				resourcePath+"disciplinary_model_0321",
				resourcePath+"stopwords.txt",
				resourcePath+"disciplinary_model_0321/vocab.txt",resourcePath+"categories.txt");
		
		 //实例化预测网络类0322 
		NeuralNetworkModel neuralNetworkModel1 = new NeuralNetworkModel(
				resourcePath+"disciplinary_model_0322",
				resourcePath+"stopwords.txt",
				resourcePath+"disciplinary_model_0322/vocab.txt",resourcePath+"categories.txt");
		
		String test_txt = "违纪问题通报 2018年5月，浙江省嘉兴市秀洲区纪委就该区新城街道党工委原书记李荣伟履行全面从严治党主体责任不力、违反党的工作纪律问题进行通报。 经查，2012年2月至2016年12月，李荣伟（现任嘉兴市秀禾农业投资集团有限公司党委副书记）在担任嘉兴市秀洲区新城街道党工委书记期间，作为全面从严治党第一责任人，未对重点领域和关键岗位强化监管，疏于对党员干部的教育、管理和监督，对有违纪违法行为的干部未及时进行组织处理，在权力制约、预防腐败工作上履职不到位，履行全面从严治党主体责任不力，2017年12月，经嘉兴市秀洲区纪委常委会研究，决定给予李荣伟党内警告处分。 ●事件回顾 2016年4月，接群众举报，浙江省嘉兴市秀洲区纪委对新城街道办事处原副主任陈加明进行纪律审查；8月，陈加明受到开除党籍和开除处分，并被移送检察机关处理。 “拔出萝卜带出泥”。随后，新城街道创建办原主任万炳炎、城建办原副主任王荣根、社会事业所聘用人员朱培明等3人也因严重违纪问题接受区纪委的纪律审查。 经查，陈加明等4人的违纪违法行为主要发生在李荣伟担任党工委书记期间。其中，陈加明单独或伙同万炳炎、王荣根、朱培明等人，索取或收受他人财物共计价值64.38万元；万炳炎单独或伙同陈加明等人，索要或收受他人贿赂共计价值27.3万元；王荣根伙同陈加明等人索要或收受他人贿赂共计价值17万元；朱培明伙同陈加明收受贿赂共计价值24.9万元。2017年1月至10月，陈加明、万炳炎、朱培明、王荣根等4人因犯受贿罪先后被判刑。 在李荣伟担任书记的5年时间里，为什么会发生陈加明等4人的违纪违法问题？李荣伟在履行全面从严治党主体责任方面是否存在不作为等情况？对此，嘉兴市秀洲区纪委展开了调查。 ●查处经过 在对新城街道4起案件进行纪律审查时，调查人员发现了李荣伟在任职期间存在履行全面从严治党主体责任不力的问题线索。在经过缜密细致的初步核实后，2017年10月23日，区纪委决定对李荣伟涉嫌违纪问题予以立案审查。 随着调查的深入，李荣伟履行全面从严治党主体责任不力的违纪事实逐渐浮出水面…… 2015年上半年，新城街道对九里村土地综合整治工程项目进行发包，涉及工程量100多万元，中标单位是嘉兴市某建设有限公司，而这家公司的主要股东之一陈某恰好是陈加明的儿子；而该公司的技术负责人则是时任街道农技中心主任的万炳炎。 对于这么明显存在利益冲突的情况，当时作为该街道党工委书记的李荣伟没有进行深入调查了解，也没有要求街道纪工委对反映陈加明儿子参股的情况深入调查。据李荣伟交代，他主要还是碍于情面，当时整治项目的征迁工作任务重、难度大，又没有得力人选可以替代，所以就迁就他们，在大是大非问题上决断力、处置力不够。 2016年5月，时任新城街道城建办副主任的王荣根因涉嫌违纪问题向区纪委投案自首。直到2017年4月6日，王荣根才被街道党工委免去职务。 “当时我作为街道的一把手，对王荣根去投案的事情是清楚的，但这件事没有引起我的重视，未及时对其进行组织处理，导致其在初核调查期间仍然担任街道中层职务，这件事对整个街道工作氛围及选人用人导向都产生了不良影响。”李荣伟说。 经查，李荣伟身为党工委主要负责人、全面从严治党第一责任人，没有亲自部署和过问党风廉政建设工作，没有全面掌握街道的党风廉政建设情况，对于群众的信访举报也没有认真进行调查处置，特别是对陈加明、万炳炎等人出现了一些倾向性、苗头性的问题之后，没有进行相应的约谈，没有履行提醒告诫和加强监督约束的责任。4名下属严重违纪违法且构成犯罪的问题发生，与李荣伟的履职不到位有关，并造成严重不良影响。 “在作为‘班长’期间发生了系列腐败案件，我作为第一责任人肯定是脱不了干系的，在这个问题上我愿接受组织的任何处理……”李荣伟表示。 2017年12月，经嘉兴市秀洲区纪委常委会研究，决定给予李荣伟党内警告处分。";
		//String test_txt = "公款旅游";
		//调用预测方法获取违纪行为类别
		Map<String, Object>result = neuralNetworkModel.getPredictResult(test_txt,400);
		//Map<String, Object>result = neuralNetworkModel1.getPredictResult(test_txt,600);
		List<Map<String, Object>>results = (List<Map<String, Object>>) result.get("result");
		for(Map<String, Object> r:results) {
			System.out.println("违纪行为:"+r.get("label").toString()+",概率:"+r.get("prob").toString());
		}
		
	 }

}

class NeuralNetworkModel{
	/**
	 * 神经网络模型
	 */
	private SavedModelBundle model;
	/**
	 * tensorflow会话
	 */
	private Session tfSession;
	/**
	 * 停用词
	 */
	private List<String>stopwords;
	/**
	 * 词与词id的映射
	 */
	private Map<String, Object>word2id;
	/**
	 * 标签索引与标签的映射
	 */
	private Map<Integer, String>id2label;
	/**
	 * 构建网络类
	 * @param modelPath 模型加载路径
	 * @param stopwordsPath 停用词表路径
	 * @param vocabPath 词汇表路径
 	 * @param labelPath 标签索引路径
	 * @throws IOException
	 * @throws FileNotFoundException
	 */
	public NeuralNetworkModel(String modelPath,String stopwordsPath,String vocabPath,String labelPath) throws IOException, FileNotFoundException {
		model = SavedModelBundle.load(modelPath,"serve");
		tfSession = model.session();
		stopwords = new ArrayList<String>();
		word2id = new HashMap<String, Object>();
		id2label = new HashMap<Integer, String>();
		try {//构建停用词表
			BufferedReader br = new BufferedReader(new FileReader(new File(stopwordsPath)));
			String txt = "";
			while ((txt=br.readLine())!=null) {
				//System.out.println(txt.trim());
				stopwords.add(txt.trim());
			}
			br.close();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
		try {//构建词与词id的映射
			BufferedReader br = new BufferedReader(new FileReader(new File(vocabPath)));
			int id = 0;
			String txt = "";
			while((txt=br.readLine())!=null) {
				word2id.put(txt.trim(), id);
				id++;
			}
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
		try {//构建标签索引与标签的映射
			BufferedReader br = new BufferedReader(new FileReader(new File(labelPath)));
			int id = 0;
			String txt = "";
			while((txt=br.readLine())!=null) {
				id2label.put(id,txt.trim());
				id++;
			}
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
		
	}
	public SavedModelBundle getModel(){
		return model;
	}
	
	public Session getTfSession() {
		return tfSession;
	}
	
	public List<String>getStopwords(){
		return stopwords;
	}
	
	/**
	 * 预测违纪行为
	 * @param content 输入文本内容
	 * @param txt_length 模型要求的文本最大长度(0321要求400,0322要求600)
	 * @return 标签预测结果
	 */
	public Map<String,Object>getPredictResult(String content,int txt_length){
		//分词
		List<String>words = Segmenter.segment(content);
		//System.out.println(words);
		//去停用词并将词汇转化成词id
		List<Integer>inputIds = new ArrayList<Integer>();
		for(int i=0;i<words.size();i++) {
			if((i<txt_length)&&(!this.stopwords.contains(words.get(i)))&&word2id.containsKey(words.get(i))) {
				inputIds.add((Integer) word2id.get(words.get(i)));
			}
		}
		int real_length = inputIds.size();
		
		int[][] inputs = new int[1][txt_length];
		for(int i=0;i<real_length;i++) {
			inputs[0][i] = inputIds.get(i);
		}
		//如果长度不足固定长度(400),则补0
		for(int i=0;i<txt_length-real_length;i++) {
			inputs[0][i+real_length] = (Integer) word2id.get("<PAD>");
		}
		
		Tensor input_x = Tensor.create(inputs);
		Tensor keep_prob = Tensor.create(1.0f);
		//将数据输入模型,获取计算结果(标签索引)
		Tensor<?> out = tfSession.runner().feed("input_x", input_x).feed("keep_prob",keep_prob).fetch("score/Sigmoid").run().get(0);
		float[][] preds = new float[1][420];
	    out.copyTo(preds);
	    float[] pred = preds[0];
	    List<Map<String, Object>>results = new ArrayList<Map<String, Object>>();
	    float max_pred = 0;
	    int max_index = 0;
	    
	    for(int i=0;i<pred.length;i++) {
	    	if(pred[i]>=0.5) {//获取概率超过0.5的类别
	    		Map<String, Object>map = new HashMap<String, Object>();
	    		map.put("label", this.id2label.get(i));//类别
	    		map.put("prob", pred[i]);//相应概率
	    		results.add(map);
	    	}
	    	if(pred[i]>max_pred) {
	    		max_pred = pred[i];
	    		max_index = i;
	    	}
	    }
	    if(results.size()<1) {//如果都概率没超过0.5,则取最大概率对应类别
	    	Map<String, Object>map = new HashMap<String, Object>();
	    	map.put("label", this.id2label.get(max_index));
	    	map.put("prob", max_pred);
	    	results.add(map);
	    }
		Map<String, Object>result = new HashMap<String, Object>();
		result.put("result", results);
		return result;
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
}