<template>
    <n-menu
        :collapsed="collapsed"
        :collapsed-width="64"
        :collapsed-icon-size="22" 
        @update:value="handleUpdateValue" 
        :options="menuOptions" 
        :render-label="renderMenuLabel"
    />
</template>

<script setup>
import {h,ref} from 'vue'
import { RouterLink } from 'vue-router'
import { NMenu, NIcon, useMessage, NMessageProvider, useLoadingBar } from 'naive-ui'
import { creatMenuOption } from "../../utils/index.js";
import routes from "../../routes.js"

const menuOptions = creatMenuOption(routes)
const message = useMessage()
// let activeMenu = ref('home')
// let collapsed = true;
function handleUpdateValue() {
    // message.info("test");
}
function renderMenuLabel (option) {
    return h(
        RouterLink,
        {
            to: option.key
        },
        { default: () => option.label }
    )
}
</script>

<script>
export default {
    props: [
        "collapsed",
    ],
    data () {
        return {
            // collapsed: true,
        }
    }
}
</script>