<template>
  <n-divider dashed>原始图片预测</n-divider>
  <n-grid x-gap="12" :cols="2">
    <n-grid-item>
      <n-card ref="cardOrigin" title="图片上传 & 模型选择">
        <template #cover>
          <n-image ref="cardCover" :src="imageOriginUrl" />
        </template>
        <n-space vertical>
          <n-radio-group
            name="modelSelectGroup"
            style="margin-bottom: 6px"
            @update:value="
              (value) => {
                modelSelected = value;
              }
            "
          >
            <n-radio-button v-for="model in models" :key="model" :value="model">
              {{ model }}
            </n-radio-button>
          </n-radio-group>
          <n-button
            :loading="isPredBtnLoading"
            type="info"
            ghost
            :disabled="!fileListLength"
            @click="btnStartInference"
            style="margin-bottom: 12px"
          >
            <template #icon>
              <n-icon>
                <DiceOutline />
              </n-icon>
            </template>
            开始预测
          </n-button>
        </n-space>

        <n-upload
          accept=".png, .jpg, .jpeg"
          @change="handleUploadChange"
          :default-upload="false"
          ref="upload"
        >
          <n-button ghost type="primary">
            <template #icon>
              <n-icon>
                <AttachOutline />
              </n-icon>
            </template>
            选择图片
          </n-button>
        </n-upload>
      </n-card>
    </n-grid-item>
    <n-grid-item>
      <n-card ref="cardPred">
        <!-- <template ref="cardCover2" #cover>
          <n-image ref="cardCoverPred" :src="imagePredUrl" />
        </template> -->
        <n-space vertical>
          <n-divider dashed>预测表格</n-divider>
          <n-data-table
            :max-height="600"
            ref="tablePred"
            virtual-scroll
            :columns="tabCols"
            :data="tabData"
          />
          <n-divider dashed>直方图</n-divider>
          <n-card>
            <div ref="barChartDiv"></div>
          </n-card>
        </n-space>
      </n-card>
    </n-grid-item>
  </n-grid>
  <n-divider dashed>对抗样本算法选择</n-divider>
  <n-space vertical>
    <n-card>
      <n-space vertical>
        <!-- <n-button
        circle
        ghost
        type="info"
        color="#ff69b4"
        @click="setValidAdvMethodVisible"
      >
        <template #icon>
          <n-icon>
            <HourglassOutline />
          </n-icon>
        </template>
      </n-button> -->
        <n-select
          @update:value="setValidAdvMethodVisible"
          multiple
          :options="options"
          ref="advAlgorithmSelected"
        />
        <!-- <n-select v-model:value="value" multiple disabled :options="options" /> -->
        <n-button
          :loading="isAttackBtnLoading"
          type="info"
          ghost
          :disabled="!fileListLength || !isFgsmEanbled"
          @click="btnStartAdvAttack"
          style="margin-bottom: 12px"
        >
          <template #icon>
            <n-icon>
              <LogoAppleAr />
            </n-icon>
          </template>
          执行攻击算法
        </n-button>
      </n-space>
    </n-card>

    <n-divider dashed>对抗样本算法参数设置</n-divider>
    <n-card>
      <n-space vertical>
        <n-card v-if="isFgsmEanbled" title="FGSM">
          <n-input-number />
        </n-card>
      </n-space>
    </n-card>
  </n-space>
  <n-divider dashed>对抗样本结果分析</n-divider>
  <n-grid x-gap="12" :cols="2">
    <n-grid-item>
      <n-card ref="cardAdvImage" title="对抗样本">
        <template #cover>
          <n-image ref="cardCoverAdv" :src="imagePredUrlAdv" />
        </template>
      </n-card>
    </n-grid-item>
    <n-grid-item>
      <n-card ref="cardPredAdv">
        <n-space vertical>
          <n-divider dashed>对抗样本-预测表格</n-divider>
          <n-data-table
            :max-height="600"
            ref="tablePredAdv"
            virtual-scroll
            :columns="tabCols"
            :data="tabDataAdv"
          />
          <n-divider dashed>对抗样本-预测直方图</n-divider>
          <n-card>
            <div ref="barChartDivAdv"></div>
          </n-card>
        </n-space>
      </n-card>
    </n-grid-item>
  </n-grid>
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
  NSelect,
  NInputNumber,
  NRadioGroup,
  NRadioButton,
  NRadio,
} from "naive-ui";
import axios from "axios";
import { useMessage, useLoadingBar } from "naive-ui";
import imageNAUrl from "../assets/NA.png";
import {
  HourglassOutline,
  AlertCircleOutline,
  AccessibilityOutline,
  AttachOutline,
  DiceOutline,
  LogoAppleAr,
} from "@vicons/ionicons5";
import Plotly from "plotly.js-dist-min";
</script>

<script>
const models = ["model_resnet18", "model_resnet50"];
let options = [
  {
    label: "FGSM-based",
    value: "fgsm-based",
    disabled: true,
  },
  {
    label: "FGSM",
    value: "fgsm",
  },
  {
    label: "IFGSM",
    value: "ifgsm",
  },
  {
    label: "PIFGSM",
    value: "pi-fgsm",
  },
  {
    label: "DE-based",
    value: "de-based",
    disabled: true,
  },
  {
    label: "OnePixelAttack",
    value: "onepixel",
  },
];
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
];

// let tabData = [];
export default {
  components: {
    AccessibilityOutline,
    HourglassOutline,
    AlertCircleOutline,
  },
  data() {
    return {
      models: models,
      modelSelected: null,
      isFgsmEanbled: false,
      // advMethodsSelected: null,
      options: options,
      tabData: [],
      tabDataAdv: [],
      tabCols: tabCols,
      // pagination: {pageSize: 5},
      isPredBtnLoading: false,
      isAttackBtnLoading: false,
      fileListLength: 0,
      fileList: [],
      message: useMessage(),
      loadingBar: useLoadingBar(),
      imageOriginUrl: imageNAUrl,
      imagePredUrl: imageNAUrl,
      imagePredUrlAdv: imageNAUrl,
    };
  },
  methods: {
    setValidAdvMethodVisible(value) {
      if (value == null || value == undefined) {
        return;
      }
      this.isFgsmEanbled = value.includes("fgsm");
    },
    errLog(msg, duration) {
      this.message.error(msg, {
        // this.message.error("发生错误：\n"+ msg, {
        closable: true,
        duration: duration,
      });
      this.loadingBar.error();
    },
    handleUploadChange({ fileList }) {
      this.fileList = fileList;
      this.fileListLength = fileList.length;
      if (this.fileListLength > 1) {
        // this.message.info("抱歉啊，目前只能上传一张图，已清除上次图片");
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
    btnStartInference() {
      let imagePostUrl = null;
      if (this.modelSelected == null) {
        this.message.error("请选择模型");
        return;
      } else if (this.modelSelected == "model_resnet18") {
        imagePostUrl = "http://localhost:23333/resnet/18/prediction";
      } else if (this.modelSelected == "model_resnet50") {
        imagePostUrl = "http://localhost:23333/resnet/50/prediction";
      } else {
        this.message.error("模型错误");
        return;
      }
      let formData = new FormData();
      formData.append("file", this.fileList[0].file);
      this.loadingBar.start();
      this.isPredBtnLoading = true;
      axios
        .post(imagePostUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          transformRequest: [
            function (data, headers) {
              // console.log(data);
              return data;
            },
          ],
          // validateStatus: (status) => {
          //   return true; // I'm always returning true, you may want to do it depending on the status received
          // },
        })
        .then((response) => {
          if (response.status == 200) {
            this.message.success("预测成功");
            this.loadingBar.finish();
            // this.imagePredUrl =
            //   "http://localhost:23333/imgs/" + response.data["image_url"];
            this.fillTablePred(this.tabData, response.data);
            this.plotBarChart(this.$refs.barChartDiv, response.data);
          } else {
            // console.table(response.data);
            this.errLog("Error Code:" + response.status, 10000);
          }
        })
        .catch((e) => {
          // console.log(e.toString());
          this.errLog(e.toString(), 10000);
        })
        .finally(() => {
          this.isPredBtnLoading = false;
        });
    },
    btnStartAdvAttack() {
      let imagePostUrl = "http://localhost:23333/resnet/18/attack/fgsm";
      this.imagePredUrlAdv = this.imageNAUrl;
      // if (this.modelSelected == null) {
      //   this.message.error("请选择模型");
      //   return;
      // } else if (this.modelSelected == "model_resnet18") {
      //   imagePostUrl = "http://localhost:23333/resnet/18/prediction";
      // } else if (this.modelSelected == "model_resnet50") {
      //   imagePostUrl = "http://localhost:23333/resnet/50/prediction";
      // } else {
      //   this.message.error("模型错误");
      //   return;
      // }

      let formData = new FormData();
      formData.append("file", this.fileList[0].file);
      this.loadingBar.start();
      this.isAttackBtnLoading = true;
      axios
        .post(imagePostUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          transformRequest: [
            function (data, headers) {
              // console.log(data);
              return data;
            },
          ],
          // validateStatus: (status) => {
          //   return true; // I'm always returning true, you may want to do it depending on the status received
          // },
        })
        .then((response) => {
          if (response.status == 200) {
            this.message.success("告：攻击已完成");
            this.loadingBar.finish();
            this.imagePredUrlAdv =
              "http://localhost:23333/" + response.data["advs"];
            this.fillTablePred(this.tabDataAdv, response.data);
            this.plotBarChart(this.$refs.barChartDivAdv, response.data);
          } else {
            console.table(response.data);
            this.errLog(response.data["detail"], 10000);
          }
        })
        .catch((e) => {
          // console.log(e.toString());
          this.errLog(e.toString(), 10000);
        })
        .finally(() => {
          this.isAttackBtnLoading = false;
        });
    },
    fillTablePred(src, data) {
      const clz = data["classes"];
      const confs = data["confidences"];
      const iids = data["iids"];
      console.log(clz);
      if (!clz) {
        return;
      }
      for (let index = 0; index < clz.length; index++) {
        src.push({
          key: index,
          iid: iids[index],
          name: clz[index],
          confidence: confs[index],
        });
      }
    },
    plotBarChart(barDiv, data) {
      const clz = data["classes"];
      const confs = data["confidences"];

      let trace1 = {
        x: clz,
        y: confs,
        type: "bar",
        text: confs.map(String),

        textposition: "auto",
        hoverinfo: "none",
        marker: {
          color: "#63E2B7",
          opacity: 0.8,
          line: {
            color: "#63E2B7",
            width: 1.5,
          },
        },
      };

      let data1 = [trace1];

      let layout = {
        title: "top-10 classes & confidences",
        barmode: "stack",
        plot_bgcolor: "rgba(0, 0, 0, 0)",
        paper_bgcolor: "rgba(0, 0, 0, 0)",
        font: {
          // family: "Courier New, monospace",
          size: 18,
          color: "#7f7f7f",
        },
      };
      let config = {
        responsive: true,
      };
      Plotly.newPlot(barDiv, data1, layout, config);
    },
  },
  computed: {},
};
</script>

<style scoped>
</style>
