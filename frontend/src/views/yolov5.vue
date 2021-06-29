<template>
  <n-grid x-gap="12" :cols="2">
    <n-grid-item>
      <n-card ref="cardOrigin" title="原始图片">
        <template #cover>
          <n-image ref="cardCover" :src="imageOriginUrl" />
        </template>
        <n-button
          type="info"
          ghost
          :disabled="!fileListLength"
          @click="handleClick"
          style="margin-bottom: 12px"
        >
          开始预测
        </n-button>
        <n-upload
          accept=".png, .jpg, .jpeg"
          @change="handleChange"
          :default-upload="false"
          ref="upload"
        >
          <n-button ghost type="primary">选择图片</n-button>
        </n-upload>
      </n-card>
    </n-grid-item>
    <n-grid-item>
      <n-card ref="cardPred" title="预测结果">
        <template ref="cardCover2" #cover>
          <n-image ref="cardCoverPred" :src="imagePredUrl" />
        </template>
        <n-space vertical>
          <n-data-table
            ref="tablePred"
            :max-height="250"
            
            virtual-scroll
            :columns="tabCols"
            :data="tabData"
          />
        </n-space>
      </n-card>
    </n-grid-item>
  </n-grid>
  <n-divider />
</template>


<script setup>
import {
  NButton,
  NUpload,
  NImage,
  NCard,
  NSpace,
  NGrid,
  NGridItem,
  NDivider,
  NIcon,
  NTable,
  NDataTable,
} from "naive-ui";
import axios from "axios";
import { useMessage, useLoadingBar } from "naive-ui";
import imageNAUrl from "../assets/NA.png";
import { HourglassOutline } from "@vicons/ionicons5";
</script>

<script>
let tabCols = [
  {
    title: "键",
    key: "key",
    sorter: (row1, row2) => row1.key - row2.key,
  },
  {
    title: "名称",
    key: "name",
    defaultSortOrder: "ascend",
    sorter: "default",
  },
  {
    title: "编号",
    key: "iid",
    sorter: (row1, row2) => row1.iid - row2.iid,
  },
  {
    title: "置信度",
    key: "confidence",
    sorter: (row1, row2) => row1.confidence - row2.confidence,
  },  
  {
    title: "区域",
    key: "area",
    sorter: (row1, row2) => row1.area[0] * row1.area[1] - row2.area[0] * row2.area[1],
  },
];

let tabData = [
  // {
  //   key: 0,
  //   name: 'dog',
  //   iid: 123,
  //   confidence: 0.98,
  //   area: [123, 123]
  // },
  // {
  //   key: 1,
  //   name: 'cat',
  //   iid: 124,
  //   confidence: 0.45,
  //   area: [123, 124]
  // }
]
export default {
  data() {
    return {
      tabData: tabData,
      tabCols: tabCols,
      // pagination: {pageSize: 5},
      fileListLength: 0,
      fileList: [],
      message: useMessage(),
      loadingBar: useLoadingBar(),
      imageOriginUrl: imageNAUrl,
      imagePredUrl: imageNAUrl,
    };
  },
  methods: {

    errLog(msg, duration) {
      this.message.error(msg, {
        closable: true,
        duration: duration,
      });
      this.loadingBar.error();
    },
    handleChange({ fileList }) {
      this.fileList = fileList;
      this.fileListLength = fileList.length;
      if (this.fileListLength > 1) {
        this.message.info("抱歉啊，目前只能上传一张图，已清除上次图片");
        this.fileList.shift();
      }
      if (this.fileListLength > 0) {
        this.imageOriginUrl = URL.createObjectURL(fileList[0].file);
      } else {
        this.imageOriginUrl = imageNAUrl;
      }
      this.fileListLength = fileList.length;
      this.tabData = [];
      this.imagePredUrl = imageNAUrl;
    },
    handleClick() {
      if (this.fileListLength > 1) {
        this.message.error("抱歉啊，目前只能上传一张图，请删除多余图片。");
        return;
      }
      let formData = new FormData();
      formData.append("file", this.fileList[0].file);
      this.loadingBar.start();

      axios
        .post("http://localhost:23333/yolo/prediction", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          transformRequest: [
            function (data, headers) {
              // console.log(data);
              return data;
            },
          ],
        })
        .then((response) => {
          if (response.status == 200) {
            this.message.success("预测成功");
            this.loadingBar.finish();
            this.imagePredUrl =
              "http://localhost:23333/imgs/pred/" + response.data["image_url"];
            this.fillTablePred(response.data);
          } else {
            console.table(response.data);
            this.errLog("Error Code:" + response.status, 5000);
          }
        })
        .catch((e) => {
          console.log(e.toString());
          this.errLog(e.toString(), 30000);
        });
    },
    fillTablePred(data) {
      this.tabData = [];
      const arrBBoxes = data["boudingboxes"];
      console.log(arrBBoxes);
      if (!arrBBoxes)  {
        return;
      }
      const arrClasses = data["classes"];
      const arrConfidences = data["confidences"];
      for (let index = 0; index < arrBBoxes.length; index++) {
        const bb = arrBBoxes[index];
        const clz = arrClasses[index];
        const conf = arrConfidences[index];
        this.tabData.push({
          key: index,
          name: clz["name"],
          iid: clz["id"],
          confidence: conf,
          area: "placeholder"
        });
      }
    }
  },
};
</script>

<style scoped>
</style>
