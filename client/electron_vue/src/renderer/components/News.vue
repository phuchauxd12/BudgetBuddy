<script setup lang="ts">
import { onMounted, ref } from 'vue';
import axios from 'axios';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '../../@/components/ui/card';

const newsData = ref([]);

onMounted(async () => {
  try {
    const response = await axios.get('http://localhost:8080/api/v1/get_news');
    newsData.value = response.data.list_of_news;
    console.log('News data:', newsData.value);
  } catch (error) {
    console.error('Error fetching news:', error);
  }
});
</script>

<template>
  <div>
    <h1 class="font-semibold text-lg text-left ps-5">Latest News</h1>
    <div class="  justify-center gap-4  h-40 carousel rounded-box w-full">
    <div class="carousel-item ">
        <!-- Use v-for to loop through a range (0 to 3) and create a card for each iteration -->
      <Card
        v-for="(news, index) in newsData"
        :key="index"
        class="w-48  h-36 mt-3 me-4 relative overflow-hidden "
        :style="{ backgroundImage: `url(${news.news_image_url})`, backgroundSize: 'cover' }"
      >
        <div class="absolute inset-0 bg-black opacity-40"></div>
        <div class="relative z-10 flex flex-col"> <!-- Place text inside a relative container -->
           
            <CardHeader class=" text-ellipsis text-left h:1/2">
            <CardTitle class="text-white font-bolder text-xs overflow-hidden text-ellipsis hover:underline ">
              <a :href="news.news_url" target="_blank" rel="noopener noreferrer" class="p-0">{{news.news_title }} </a>
            </CardTitle>
            <CardDescription class="text-white text-xs">
              {{ news.update_date }}
            </CardDescription>
          </CardHeader>
          <CardContent class="text-white text-xs h:1/2">
            {{ news.publisher }}
          </CardContent>
         
        </div>
      </Card>
    </div>
    </div>
  </div>
</template>
