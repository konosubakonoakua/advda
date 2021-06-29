import { HomeOutline as HomeIcon, AppsOutline as AppsIcon, LinkOutline as LinkIcon, AppsOutline } from '@vicons/ionicons5'

import Layout from './layout/index.vue'

import { h } from 'vue'

const routes = [
  { path: '/404', component: () => import('./views/404.vue'), hidden: true },
  { path: '/login', component: () => import('./views/login.vue'), hidden: true },
  {
    path: '/',
    component: Layout,
    name: 'home',
    redirect: '/home',
    children: [
      {
        path: '/home',
        component: () => import('./views/home/home.vue'),
        name: 'home',
        label: '首页',
        icon: HomeIcon
      }
    ]
  },
  {
    path: '/examples',
    component: Layout,
    name: 'examples',
    label: '对抗样本应用实例',
    icon: AppsIcon,
    children: [
      {
        path: '/yolov5', 
        component: () => import('./views/yolov5.vue'),
        name: 'yolov5',
        label: 'yolov5 目标检测',
        icon: LinkIcon
      },
      {
        path: '/list', 
        component: () => import('./components/HelloWorld.vue'),
        name: 'list',
        label: 'helloworld',
        icon: LinkIcon
      },
      {
        path: '/other', 
        component: () => import('./views/home/home.vue'),
        name: 'other',
        label: '其他',
        icon: LinkIcon
      },
    ]
  }
]
export default routes