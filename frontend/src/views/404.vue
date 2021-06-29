<template>
<div id="div404">
  <n-result status="404" title="404 资源不存在" description="生活总归带点荒谬">
    <template #footer>
      <n-button>找点乐子吧</n-button>
    </template>
  </n-result>
</div>
</template>

<script setup>
import { NResult, NButton } from 'naive-ui'
</script>

<style scoped>
#div404 {
    position: absolute;
    left: 40%;
    top: 40%;
    /* width:200px; */
    /* height:100px; */
    /* margin-left:-100px; */
    /* margin-top:-50px; */
}
</style>