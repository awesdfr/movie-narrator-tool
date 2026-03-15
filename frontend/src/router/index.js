import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/Home.vue'),
    meta: { title: '首页' }
  },
  {
    path: '/project/:id',
    name: 'Project',
    component: () => import('@/views/Project.vue'),
    meta: { title: '项目详情' }
  },
  {
    path: '/project/:id/process',
    name: 'Process',
    component: () => import('@/views/Process.vue'),
    meta: { title: '处理' }
  },
  {
    path: '/project/:id/editor',
    name: 'Editor',
    component: () => import('@/views/Editor.vue'),
    meta: { title: '片段编辑' }
  },
  {
    path: '/project/:id/timeline',
    name: 'Timeline',
    component: () => import('@/views/Timeline.vue'),
    meta: { title: '时间轴' }
  },
  {
    path: '/project/:id/preview',
    name: 'Preview',
    component: () => import('@/views/Preview.vue'),
    meta: { title: '预览' }
  },
  {
    path: '/settings',
    name: 'Settings',
    component: () => import('@/views/Settings.vue'),
    meta: { title: '设置' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  document.title = `${to.meta.title || '页面'} - 电影解说重制工具`
  next()
})

export default router
