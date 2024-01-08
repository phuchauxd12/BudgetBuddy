<script setup lang="ts">
import Networth from '../components/Networth.vue';
import OverviewVue from '../components/Overview.vue'; 
import TransactionComponent from '../components/TransactionComponent.vue';

import Links from  '../components/Link.vue';

</script>

<template>
   <Links/>
  <section class="display grid-cols-4">
    <div class=""><Networth/></div>
    <div class="investment rounded-lg bg-gray-200 w-full col-span-3 mt-14">
   
      <div> <TransactionComponent/></div>
    </div>

    

  
   
   
  

  </section>  
</template>