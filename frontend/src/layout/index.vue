<template>
  <n-config-provider :theme="darkTheme">
  <n-loading-bar-provider>
  <n-message-provider>
  <n-layout position="absolute">
    <n-layout-header style="height: 64px; padding: 24px;" bordered>advda 对抗样本平台</n-layout-header>
    <n-layout has-sider position="absolute" style="top: 64px; bottom: 64px;">
      <n-layout-sider 
        bordered
        collapse-mode="width"
        :collapsed-width="64"
        :width="240"
        :collapsed="collapsed"
        show-trigger
        @collapse="collapsed = true"
        @expand="collapsed = false"
      >
        <vna-sidebar :collapsed="collapsed" />
      </n-layout-sider>
      <n-layout content-style="padding: 24px;">
        <router-view />
      </n-layout>
    </n-layout>
    <n-layout-footer bordered position="absolute" style="height: 64px; padding: 24px; text-align: center;">Copyright@Konosubakonoakua</n-layout-footer>
  </n-layout>
  </n-message-provider>
  </n-loading-bar-provider>
  </n-config-provider>
</template>

<script>
export default {
  data () {
    return {
      collapsed: true,
    }
  }
}
</script>

<script setup>
import { darkTheme, NConfigProvider, NMessageProvider } from 'naive-ui'
import { NLayout, NLayoutHeader, NLayoutSider, NLayoutFooter, NLoadingBarProvider } from 'naive-ui'
import VnaSidebar from './components/Sidebar.vue'
// let collapsed = true;
</script>