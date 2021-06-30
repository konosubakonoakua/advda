import { HomeOutline as HomeIcon, AppsOutline as AppsIcon, LinkOutline as LinkIcon, BugOutline } from '@vicons/ionicons5'

import Layout from './layout/index.vue'

import { h } from 'vue'

const routes = [
  { path: '/404', component: () => import('./views/404.vue'), hidden: true },
  { path: '/login', component: () => import('./views/login.vue'), hidden: true },
  {
    path: '/',
    component: Layout,
    name: 'home',
    redirect: '/detector',
    label: '菜单',
    icon: AppsIcon,
    children: [
      {
        path: '/home',
        component: () => import('./views/home/home.vue'),
        name: 'home',
        label: '首页',
        icon: HomeIcon
      },
      {
        path: '/detector', 
        component: () => import('./views/detector.vue'),
        name: 'detector',
        label: '目标检测',
        icon: BugOutline
      },
      {
        path: '/classifier', 
        component: () => import('./views/classifier.vue'),
        name: 'classifier',
        label: '分类',
        icon: BugOutline
      },
    ]
  },

  // {
  //   path: '/examples',
  //   component: Layout,
  //   name: 'examples',
  //   label: '对抗样本应用实例',
  //   icon: AppsIcon,
  //   children: [
  //     {
  //       path: '/detector', 
  //       component: () => import('./views/detector.vue'),
  //       name: 'detector',
  //       label: '目标检测',
  //       icon: LinkIcon
  //     },
  //     {
  //       path: '/classfier', 
  //       component: () => import('./components/HelloWorld.vue'),
  //       name: 'classfier',
  //       label: '分类',
  //       icon: LinkIcon
  //     },
  //     // {
  //     //   path: '/other', 
  //     //   component: () => import('./views/home/home.vue'),
  //     //   name: 'other',
  //     //   label: '其他',
  //     //   icon: LinkIcon
  //     // },
  //   ]
  // }
]
export default routes