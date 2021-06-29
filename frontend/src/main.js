import { createApp } from 'vue'
import App from './App.vue'

import router from './router'
import store from './store'
// import index from './layout/index.vue'

createApp(App).use(router).use(store).mount('#app')
