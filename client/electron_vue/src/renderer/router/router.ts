// router.ts
import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router';
import Home from '../views/Homepage.vue';
import Plan from '../views/Plan.vue';
import Login from '../views/Login.vue';
import Register from '../views/Register.vue';
import Profile from '../views/Profile.vue';
import Test from "../views/test.vue"

const routes: RouteRecordRaw[] = [
  { path: '/', component: Login },
  { path: '/profile', component: Profile},
  { path: '/home', component: Home },
  { path: '/plan', component: Plan },
  { path: '/login', component: Login },
  { path: '/register', component: Register },
  // Add more routes as needed
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;