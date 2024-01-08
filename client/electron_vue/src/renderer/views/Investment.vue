<script setup lang="ts">
import Networth from '../components/Networth.vue';
import OverviewVue from '../components/Overview.vue'; 
import InvestmentComponent from '../components/InvestmentComponent.vue';

import Links from  '../components/Link.vue';

</script>

<template>
   <Links/>
  <section class="display grid-cols-4">
    <div class=""><Networth/></div>
    <div class="investment rounded-lg bg-gray-200 w-full col-span-3 mt-14">
   
      <div><InvestmentComponent/></div>
    </div>

    

  
   
   
  

  </section>  
</template>

<style>

</style>
<!-- <script setup lang="ts">
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '../../@/components/ui/select'
</script>

<template>
  <Select>
    <SelectTrigger>
      <SelectValue placeholder="Select a fruit" />
    </SelectTrigger>
    <SelectContent>
      <SelectGroup>
        <SelectLabel>Fruits</SelectLabel>
        <SelectItem value="apple">
          Apple
        </SelectItem>
      </SelectGroup>
    </SelectContent>
  </Select>
</template> -->