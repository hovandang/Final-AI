import time
from skimage import transform
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
import cv2
from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import os
import splitwav
import librosa
import librosa.display as dsp
from IPython.display import Audio
import matplotlib.pyplot as plt
import numpy as np
from os import path
from pydub import AudioSegment
value = [' đàn t"rưng',' đàn nhị',' đàn tranh',' sáo mèo',' tiếng mõ',' cồng chiêng',' đàn bầu',' khèn']
def save_file(sound_file):
    # save your sound file in the right folder by following the path
    with open(os.path.join('./test1/', sound_file.name),'wb') as f:
         f.write(sound_file.getbuffer())
    return sound_file.name

def classification(file):
    model = load_model("model_music_instrument1.h5")
    folder = './test1'
    sample,sample_rate = librosa.load(folder+'/'+file,sr = None)
    k = librosa.get_duration(sample)
    split_wav = splitwav.SplitWavAudioMubin(folder, file)
    split_wav.multiple_split(min_per_split=5)
    i = 0
    sum  = [0,0,0,0,0,0,0,0]
    count = 0
    while ((i+15)<k):
        src = folder + '/'+ str(i)+'_' + file
        sample,sample_rate = librosa.load(src,sr = None)
        d = librosa.stft(sample)
        D = librosa.amplitude_to_db(np.abs(d),ref=np.max)
        dsp.specshow(D,y_axis='log',x_axis='s',sr=sample_rate)
        plt.axis('off')
        plt.savefig(src.rstrip('.wav')+'.png', dpi=200, bbox_inches='tight',transparent=True, pad_inches=0)
        # image path
        img_path = src.rstrip('.wav')+'.png'
        # img_path = "./2/dan_bau14.png"   
        print (img_path)
        # load a single 
        new_image = load(img_path)
        
        # check prediction
        pred = model.predict(new_image)
        # sum[0] = sum[0]+pred[0][0]
        # sum[1] = sum[1] + pred[0][1]
        # sum[2] = sum[2] +pred[0][2]
        for j in range(0,len(sum)):
            sum[j] = sum[j]+ pred[0][j]
        i = i+5
        count = count + 1
    for f in os.listdir('./test1'):
        os.remove(os.path.join('./test1', f))
    max = 0
    print(len(sum))
    for j in range(0,len(sum)):
        print(sum[j])
        if(sum[max]<sum[j]):
            max = j
    st.write('Đây là'+ value[max])
def choice_prediction():
    st.write('# Prediction')

    st.write('### Choose a marine mammal sound file in .wav format')
    # upload sound
    uploaded_file = st.file_uploader(' ', type='wav')
    if uploaded_file is not None:  
        # view details
        file_details = {'filename':uploaded_file.name, 'filetype':uploaded_file.type, 'filesize':uploaded_file.size}
        st.write(file_details)
        # read and play the audio file
        st.write('### Play audio')
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
        # save_file function
        save_file(uploaded_file)
        # define the filename
        sound = uploaded_file.name
        # transform_wav_to_csv function
        st.write('### Classification results')
        # if you select the predict button
        if st.button('Predict'):
            # write the prediction: the prediction of the last sound sent corresponds to the first column
            classification(sound)
            # st.write("The marine mammal is: ",  str(classification(transform_wav_to_csv(sound))).replace('[', '').replace(']', '').replace("'", '').replace('"', ''))
        else:
            st.write('The file has not been uploaded yet')

# def load_image(img_path, show=False):

#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_batch = np.expand_dims(img_array, axis=0)        # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)                                    # imshow expects values in the range [0, 1]
#     img_preprocessed = preprocess_input(img_batch)
#     return img_preprocessed
def load(filename):
    np_image = Image.open(filename).convert('RGB')
    np_image = np.array(np_image).astype('float32')/255.0

    np_image = transform.resize(np_image, (288, 288, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image
selected = option_menu(None, ["Diagnostic", "More","Author"], 
    icons=[ "upload", 'bookmark-fill','house'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"}
    }
)
if selected == "Author":
    st.title("Traditional vietnamese music instrument detection ")
    st.subheader("Hồ Văn Đằng 20134013")
    st.write("""**Viêm phổi** 
     là một bệnh nhiễm trùng phổi, có thể do vi khuẩn hoặc virus gây ra. May mắn thay, bệnh truyền nhiễm do vi khuẩn hoặc virus này có thể được điều trị tốt bằng thuốc kháng sinh và thuốc kháng virus. Tuy nhiên, việc phát hiện sớm và điều trị vẫn là phương pháp hữu hiệu nhất để ngăn ngừa bệnh viêm phổi. Chụp X-quang ngực hiện nay đang là phương pháp tốt nhất để chẩn đoán bệnh viêm phổi. Tuy nhiên, hình ảnh X-quang của viêm phổi thường không rõ ràng lắm và dễ bị nhầm sang các bệnh khác. Hơn nữa, hình ảnh viêm phổi do vi khuẩn hoặc virus đôi khi bị các chuyên gia phân loại nhầm dẫn đến việc cấp sai thuốc cho bệnh nhân và từ đó làm tình trạng bệnh nhân trở nên trầm trọng hơn. 
Ngoài ra lực lượng bác sĩ X quang được đào tạo ở các nước còn thấp (LRC), đặc biệt là ở các vùng nông thôn. Do đó, nhu cầu cấp thiết về các hệ thống chẩn đoán có sự hỗ trợ của máy tính (CAD), có thể giúp các bác sĩ X quang phát hiện viêm phổi từ hình ảnh X-quang ngực ngay sau khi thu nhận là một điều rất cần thiết.  Hiện nay nhiều biến chứng y sinh (ví dụ như phát hiện khối u não, phát hiện ung thư vú, v.v.) đang sử dụng các giải pháp dựa trên Trí tuệ nhân tạo (AI). Trong số các kỹ thuật học sâu, mạng nơ-ron tích tụ (CNN) đã cho thấy nhiều hứa hẹn trong việc phân loại hình ảnh và do đó được cộng đồng nghiên cứu áp dụng rộng rãi. Các kỹ thuật học máy sâu trên X- quang ngực đang trở nên phổ biến vì chúng có thể dễ dàng sử dụng với kỹ thuật hình ảnh chi phí thấp và có nhiều dữ liệu có sẵn để đào tạo các mô hình học máy khác nhau. Đó cũng là lí do em chọn đề tài: **“Nhận diện bệnh viêm phổi bằng ảnh x- ray, đưa ra các chỉ dẫn lâm sàn”**
""")
    st.write("Với các datasets là các hình ảnh được lấy từ kagle, thông qua Google Colab, em tạo ra được một model trainning với độ chính xác là 93%. Các dữ liệu model training sẽ được đưa lên webApp Streamlit, để thao tác cũng như hiển thị kết quả một cách dễ dàng hơn.")
    st.write("Do thời lượng cũng như kiến thức còn hạn hẹp nên trong quá trình tạo dựng mô hình còn nhiều sai sót, mong thầy và các bạn thông cảm và góp ý.")
    
if selected == "Diagnostic":
    
    choice_prediction()

    # ### load file
    # uploaded_file = st.file_uploader("Insert File")

    # map_dict = {0: 'NORMAL',
    #             1: 'PNEUMONIA'}
    
    # doan = {0: 'BACTERIAL PNEUMONIA',
    #         1: 'VIRAL PNEUMONIA'}
    
    # if uploaded_file is not None:
    #     # Convert the file
    #     # imga = image.load_img(uploaded_file,target_size=(256,256))
    #     # st.image(imga, channels="RGB")
    #     # img = image.load_img(uploaded_file,target_size=(64,64))
    #     #st.image(uploaded_file, channels="RGB")
    #     #plt.imshow(img)
    #     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    #     image = cv2.imdecode(file_bytes, 1)
    #     img = cv2.resize(image,(224,224))     # resize image to match model's expected sizing
    #     img = img.reshape(1,224,224,3) 
    #     img = img.astype('float32')
    #     img = img/255
    #     #Button
    #     Genrate_pred = st.button("Generate Prediction") 
    
    #     if Genrate_pred:
    
    #         with st.spinner("Running!"):
    #             time.sleep(2)
    #         prediction = model.predict(img).argmax()
    #         y_pred = model.predict(img)
    #         y_pre = model.predict(img)
    #         if prediction == 0:
    #             st.write("**Predicted Label for the image is NORMAL**")
    #             a = y_pred.max()
    #             a = a*100
    #             st.write("**Accuracy:** ",a,"%")
    #         else:
    #             prediction1 = model1.predict(img).argmax()
    #             st.write("**Predicted Label for the image is {}**".format(doan [prediction1]))
    #             b = y_pre.max()
    #             b = b*100
    #             st.write("**Accuracy:** ", b,"%")
  
if selected == "More":
    
    st.title("BỆNH VIÊM PHỔI: NGUYÊN NHÂN, TRIỆU CHỨNG, ĐIỀU TRỊ VÀ CÁCH PHÒNG NGỪA")        
    st.title("------------------------------------------------")
    st.write("**Viêm phổi** đang gây ra gánh nặng bệnh tật lớn đặc biệt với sự xuất hiện của Covid-19, di chứng viêm phổi hậu Covid-19 cũng đang là nguy cơ đe dọa nghiêm trọng sức khỏe, gây suy hô hấp đối với nhiều bệnh nhân F0 hiện nay.")
    st.header("Viêm phổi là gì?")
    st.write("**Viêm phổi** là tình trạng viêm nhiễm (sưng) nhu mô phổi bao gồm viêm phế nang (túi khí nhỏ), túi phế nang, ống phế nang, tổ chức liên kết khe kẽ và viêm tiểu phế quản tận cùng, chủ yếu do vi khuẩn, virus, nấm gây nên. Các phế nang, đường dẫn khí chứa nhiều dịch nhầy hoặc mủ, xuất tiết dịch đường hô hấp trên gây ho đờm, sốt ớn lạnh, khó thở. Hiện tượng viêm phổi có thể ở một vùng hoặc ở một vài vùng (viêm phổi thùy hoặc “đa thùy”) hoặc toàn bộ phổi.")
    st.header("Nguyên nhân gây bệnh viêm phổi")
    st.subheader("1. Viêm phổi mắc phải cộng đồng")
    st.write("""**+ Viêm phổi do vi khuẩn**": "**Vi khuẩn** là nguyên nhân thường gặp nhất gây viêm phổi ở trẻ em và người lớn. Viêm phổi do vi khuẩn nếu không nhận biết sớm và điều trị kịp thời sẽ dễ dẫn đến hậu quả khó lường, thậm chí tử vong. Các loại vi khuẩn thường gặp gồm: Streptococcus pneumoniae, Legionella pneumophila, Haemophilus influenzae, Mycoplasma pneumoniae, Chlamydia pneumoniae,…""")
    st.write("""**+ Viêm phổi do virus (bao gồm Covid-19)**: Hiện nay, virus SARS-CoV-2 là tác nhân nguy hiểm nhất gây viêm phổi, trong đó, virus có thể làm hỏng phế nang và khiến chất lỏng tích tụ trong phổi. Điều đó cũng có thể dẫn đến sự phát triển của hội chứng suy hô hấp cấp tính (ARDS) – một dạng suy hô hấp nghiêm trọng.""")
    st.write("""**+ Viêm phổi do nấm**:"Viêm phổi do nấm là tình trạng bệnh nhân hít phải bào tử của nấm gây viêm nhiễm, ảnh hưởng nghiêm trọng đến hệ hô hấp với mức độ phát triển nhanh. Bệnh thường có diễn biến phức tạp nếu không được điều trị kịp thời có thể gây ra những biến chứng nguy hiểm thậm chí thiệt mạng.""")
    st.write("""**+ Viêm phổi do hóa chất**:
             Viêm phổi do hóa chất thường hiếm gặp, ít xảy ra nhưng cực kỳ nguy hiểm vì tỷ lệ gây tử vong cao cho người bệnh. Tùy thuộc vào loại hóa chất đã phơi nhiễm mà có mức độ nguy hiểm khác nhau. Bên cạnh tổn thương phổi, các hóa chất còn có thể gây hại cho nhiều cơ quan khác. Chính vì vậy việc phòng ngừa các tác nhân gây bệnh viêm phổi là việc làm cấp thiết, tránh những hậu quả đáng tiếc xảy ra.""")
    st.subheader("2. Viêm phổi mắc phải ở bệnh viện")
    st.write("Theo nghiên cứu, viêm phổi bệnh viện ở các nước phát triển chiếm tỷ lệ 15% trong tổng số trường hợp nhiễm khuẩn bệnh viện và chiếm 27% xảy ra nhiễm khuẩn này ở khoa hồi sức cấp cứu. Những vi khuẩn gây ra tình trạng này là P. aeruginosa, Acinetobacter spp, Enterobacteriacae, Haemophillus spp, S. aureus (MRSA), Streptococcus spp,…")
    st.write("Tại Việt Nam, viêm phổi bệnh viện chiếm tỷ lệ khoảng từ 21% đến 75%, trong đó viêm phổi do lây nhiễm qua thở máy chiếm đến 90% và được xác định sau thở máy 48 giờ. Đây là một vấn đề rất khó khăn mà các khoa lâm sàng, đặc biệt là khoa hồi sức tích cực phải đương đầu vì khó chẩn đoán, kéo dài thời gian điều trị, tốn kém rất nhiều chi phí.")
    st.subheader("3. Viêm phổi do hít thở")
    st.write("**Nguyên nhân gây viêm phổi** ở trường hợp này là do người bệnh hít phải lượng lớn dị vật từ đường thở (miệng, hầu họng, dạ dày,…) và dị vật rơi vào phổi 2 bên. Các dị vật hít phải có thể là nước bọt, thức ăn, hóa chất, axit dịch vị,… nếu chúng đi vào phổi sẽ kích thích phản ứng viêm của niêm mạc phổi, tạo cơ hội để vi khuẩn xâm nhập và gây viêm phổi.")
    st.header("Thời gian ủ bệnh viêm phổi bao lâu?")
    st.write("Phần lớn trường hợp, viêm phổi thường xuất hiện ở dạng cấp tính (bệnh kéo dài dưới 6 tuần) với các triệu chứng rất rõ ràng ở những ngày đầu mới phát bệnh. Đặc biệt, nếu tình trạng khó thở càng trở nặng thì nguy cơ tử vong trong thời gian ngắn càng cao.")
    st.write("Mặt khác, viêm phổi mạn tính cũng có biểu hiện tương tự, thời gian bệnh kéo dài không dứt. Một người được chẩn đoán mắc viêm phổi mạn tính khi bệnh kéo dài quá 6 tuần.")
    st.header("Dấu hiệu viêm phổi thường gặp")
    st.write("Các triệu chứng viêm phổi thường gặp như: Đau ngực khi thở hoặc ho; Ho, ho khan, ho có đờm; Sốt trên 38 độ, đổ mồ hôi và ớn lạnh; Mệt mỏi, uể oải và chán ăn; Thở nhanh, khó thở khi gắng sức; Buồn nôn, nôn mửa hoặc tiêu chảy;")
    st.title("Để biết thêm thông tin chi tiết: https://vnvc.vn/benh-viem-phoi-nguyen-nhan-trieu-chung-dieu-tri-va-cach-phong-ngua/")