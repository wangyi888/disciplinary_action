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
		 System.out.println(resourcePath+"disciplinary_model_0322");
		 //实例化预测网络类
		NeuralNetworkModel neuralNetworkModel = new NeuralNetworkModel(
				resourcePath+"disciplinary_model_0322",
				resourcePath+"stopwords.txt",
				resourcePath+"vocab.txt",resourcePath+"categories.txt");
		String test_txt = "北京市昌平区崔村镇西辛峰村党支部组织党员公款旅游。2013年6月，西辛峰村党支部组织部分党员赴大连参观旅游，并给未参加的党员发放购物卡。昌平区纪委给予西辛峰村党支部书记王德全党内警告处分";
		//调用预测方法获取违纪行为类别
		Map<String, Object>result = neuralNetworkModel.getPredictResult(test_txt);
		System.out.println(result.get("result"));
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
	 * @return 标签预测结果
	 */
	public Map<String,Object>getPredictResult(String content){
		//分词
		List<String>words = Segmenter.segment(content);
		//System.out.println(words);
		//去停用词并将词汇转化成词id
		List<Integer>inputIds = new ArrayList<Integer>();
		for(int i=0;i<words.size();i++) {
			if((i<400)&&(!this.stopwords.contains(words.get(i)))&&word2id.containsKey(words.get(i))) {
				inputIds.add((Integer) word2id.get(words.get(i)));
			}
		}
		int real_length = inputIds.size();
		
		int[][] inputs = new int[1][400];
		for(int i=0;i<real_length;i++) {
			inputs[0][i] = inputIds.get(i);
		}
		//如果长度不足固定长度(400),则补0
		for(int i=0;i<400-real_length;i++) {
			inputs[0][i+real_length] = (Integer) word2id.get("<PAD>");
		}
		
		Tensor input_x = Tensor.create(inputs);
		Tensor keep_prob = Tensor.create(1.0f);
		//将数据输入模型,获取计算结果(标签索引)
		Tensor<?> out = tfSession.runner().feed("input_x", input_x).feed("keep_prob",keep_prob).fetch("score/Sigmoid").run().get(0);
		float[][] preds = new float[1][420];
	    out.copyTo(preds);
	    float[] pred = preds[0];
	    List<String>labels = new ArrayList<String>();
	    float max_pred = 0;
	    int max_index = 0;
	    for(int i=0;i<pred.length;i++) {
	    	if(pred[i]>=0.5) {//获取概率超过0.5的类别
	    		labels.add(this.id2label.get(i));
	    	}
	    	if(pred[i]>max_pred) {
	    		max_pred = pred[i];
	    		max_index = i;
	    	}
	    }
	    if(labels.size()<1) {//如果都概率没超过0.5,则取最大概率对应类别
	    	labels.add(this.id2label.get(max_index));
	    }
		Map<String, Object>result = new HashMap<String, Object>();
		result.put("result", labels);
		return result;
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
}