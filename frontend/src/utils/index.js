import {h} from 'vue'
import { NIcon } from 'naive-ui'

function renderIcon (icon) {
  return () => h(NIcon, null, { default: () => h(icon) })
}

export function creatMenuOption(routes) {
  let res = []
  routes.forEach(route => {
    if (route.hidden) {
      return
    }
    let newOption
    if (route.children && route.children.length === 1) {
      const { path, name, label, icon } = route.children[0]
      newOption = {key: path, label }
      if(icon){
        newOption.icon = renderIcon(icon)
      }
   } else {
      const { path, name, label, icon } = route
      newOption = {key: path, label }
      if(icon){
        newOption.icon = renderIcon(icon)
      }
      if(route.children){
        const children = creatMenuOption(route.children)
        newOption.children = children
      }
   }
   res.push(newOption)
  })
  return res
}