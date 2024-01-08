// router.ts
import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router';
import Home from '../views/Homepage.vue';
import Plan from '../views/Plan.vue';
import Login from '../views/Login.vue';
import Register from '../views/Register.vue';
import Profile from '../views/Profile.vue';
import Investment from '../views/Investment.vue';
import Report from '../views/Report.vue';
import Transaction from '../views/Transaction.vue';

const routes: RouteRecordRaw[] = [
  { path: '/', component: Login },
  { path: '/home',name: 'Home', component: Home , meta: { requiresAuth: true }},
  { path: '/plan',name: 'Plan', component: Plan , meta: { requiresAuth: true }},
  { path: '/login',name: 'Login', component: Login },
  { path: '/register', name: 'Register',component: Register },
  { path: '/investment',name: 'Investment', component: Investment, meta: { requiresAuth: true }},
  { path: '/report',name:'Report', component: Report , meta: { requiresAuth: true }},
  { path: '/transaction',name:'Transaction', component: Transaction , meta: { requiresAuth: true }},
  { path: '/profile', name:'Profile',component: Profile ,meta: { requiresAuth: true }},
  // Add more routes as needed
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
