<template>
    <div>
      <router-link to="/home" class="text-white" :class="{ 'font-bold': $route.name === 'Home' }">Overview</router-link>
      <router-link to="/transaction" class="text-white" :class="{ 'font-bold': $route.name === 'Transaction' }">Transaction</router-link>
      <router-link to="/plan" class="text-white" :class="{ 'font-bold': $route.name === 'Plan' }">Plan</router-link>
      <!-- Example router-link -->
      <router-link :to="{ name: 'Investment' }" class="text-white" :class="{ 'font-bold': $route.name === 'Investment' }">Investment</router-link>
      <router-link :to="{ name: 'Report' }" class="text-white" :class="{ 'font-bold': $route.name === 'Report' }">Report</router-link>
      <router-view />
    </div>
  </template>
  