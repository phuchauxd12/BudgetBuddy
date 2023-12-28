<script setup lang="ts">
import { ref, computed, DefineComponent, ComponentOptionsMixin, ExtractPropTypes } from 'vue'
import Home from './views/Homepage.vue' 
import Plan from './views/Plan.vue'
window.electronAPI.sendMessage('Hello from App.vue!');

type PublicProps = {
  title?: string;
  description?: string;
};

interface RouteDefinition {
  [path: string]: DefineComponent<{}, {}, {}, {}, {}, ComponentOptionsMixin, ComponentOptionsMixin, {}, string, PublicProps, Readonly<ExtractPropTypes<{}>>, {}>;
}

const routes: RouteDefinition = {
  '/home': Home,
  '/plan': Plan
}

const currentPath = ref(window.location.hash)

window.addEventListener('hashchange', () => {
  currentPath.value = window.location.hash
})

const currentView = computed(() => {
  return routes[currentPath.value.slice(1) || '/'];
});
</script>

<template>
  <div class="homebar">
    <ul>
        <li><a href="#/home">Overview</a></li>
        <li><a href="#transaction">Transaction</a></li>
        <li><a href="#investment">Investment</a></li>
        <li><a href="#/plan">Plan</a></li>
        <li><a href="#report">Report</a></li>
    </ul>
  </div>
  <component :is="currentView" />
</template>


