<ul class="lqd-ext-chatbot-history-list flex flex-col gap-2">
    <template
        x-for="(chatItem, index) in chatsList"
        x-show="chatsList.length"
    >
        <li
            class="lqd-ext-chatbot-history-list-item group/chat-item relative rounded-xl px-5 py-3.5 text-heading-foreground transition-colors hover:bg-heading-foreground/5 [&.lqd-active]:bg-heading-foreground/5"
            :class="{ 'lqd-active': activeChat ? activeChat == chatItem.id : index === 0 }"
        >

            <div class="flex items-start gap-3">
				<template x-if="chatItem?.chatbot_channel === 'frame'">
					<svg
						class="flex-shrink-0"
						width="32"
						height="31"
						viewBox="0 0 32 31"
						fill="none"
						xmlns="http://www.w3.org/2000/svg"
					>
						<path
							d="M16 0C7.44048 0 0.5 6.93943 0.5 15.5C0.5 24.0606 7.4398 31 16 31C24.5609 31 31.5 24.0606 31.5 15.5C31.5 6.93943 24.5609 0 16 0ZM16 4.63468C18.8323 4.63468 21.1274 6.93057 21.1274 9.76163C21.1274 12.5934 18.8323 14.8886 16 14.8886C13.1691 14.8886 10.874 12.5934 10.874 9.76163C10.874 6.93057 13.1691 4.63468 16 4.63468ZM15.9966 26.9475C13.1718 26.9475 10.5846 25.9187 8.58906 24.2158C8.10294 23.8012 7.82243 23.1931 7.82243 22.5552C7.82243 19.6839 10.1461 17.386 13.0179 17.386H18.9834C21.8559 17.386 24.1708 19.6839 24.1708 22.5552C24.1708 23.1938 23.8916 23.8005 23.4048 24.2151C21.41 25.9187 18.8221 26.9475 15.9966 26.9475Z"
							:fill="chatItem.color"
						/>
					</svg>
				</template>

				<template x-if="chatItem?.chatbot_channel === 'whatsapp'">
					<svg width="32" height="31" viewBox="0 0 32 31" fill="none" xmlns="http://www.w3.org/2000/svg">
						<rect x="0.5" width="31" height="31" rx="15.5" fill="#25D366"/>
						<path d="M21.7891 9.63672C23.3477 11.1953 24.3125 13.2363 24.3125 15.4629C24.3125 19.9902 20.5273 23.7012 15.9629 23.7012C14.5898 23.7012 13.2539 23.3301 12.0293 22.6992L7.6875 23.8125L8.83789 19.5449C8.13281 18.3203 7.72461 16.9102 7.72461 15.4258C7.72461 10.8984 11.4355 7.1875 15.9629 7.1875C18.1895 7.1875 20.2676 8.07812 21.7891 9.63672ZM15.9629 22.291C19.748 22.291 22.9023 19.2109 22.9023 15.4629C22.9023 13.6074 22.123 11.9004 20.8242 10.6016C19.5254 9.30273 17.8184 8.59766 16 8.59766C12.2148 8.59766 9.13477 11.6777 9.13477 15.4258C9.13477 16.7246 9.50586 17.9863 10.1738 19.0996L10.3594 19.3594L9.6543 21.8828L12.252 21.1777L12.4746 21.3262C13.5508 21.957 14.7383 22.291 15.9629 22.291ZM19.748 17.1699C19.9336 17.2812 20.082 17.3184 20.1191 17.4297C20.1934 17.5039 20.1934 17.9121 20.0078 18.3945C19.8223 18.877 19.0059 19.3223 18.6348 19.3594C17.9668 19.4707 17.4473 19.4336 16.1484 18.8398C14.0703 17.9492 12.7344 15.8711 12.623 15.7598C12.5117 15.6113 11.8066 14.6465 11.8066 13.6074C11.8066 12.6055 12.3262 12.123 12.5117 11.9004C12.6973 11.6777 12.9199 11.6406 13.0684 11.6406C13.1797 11.6406 13.3281 11.6406 13.4395 11.6406C13.5879 11.6406 13.7363 11.6035 13.9219 12.0117C14.0703 12.4199 14.5156 13.4219 14.5527 13.5332C14.5898 13.6445 14.627 13.7559 14.5527 13.9043C14.1816 14.6836 13.7363 14.6465 13.959 15.0176C14.7754 16.3906 15.5547 16.873 16.7793 17.4668C16.9648 17.5781 17.0762 17.541 17.2246 17.4297C17.3359 17.2812 17.7441 16.7988 17.8555 16.6133C18.0039 16.3906 18.1523 16.4277 18.3379 16.502C18.5234 16.5762 19.5254 17.0586 19.748 17.1699Z" fill="white"/>
					</svg>
				</template>

				<template x-if="chatItem?.chatbot_channel === 'telegram'">
					<svg id="Livello_1" width="32" height="31" data-name="Livello 1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 240 240"><defs><linearGradient id="linear-gradient" x1="120" y1="240" x2="120" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#1d93d2"/><stop offset="1" stop-color="#38b0e3"/></linearGradient></defs><title>Telegram_logo</title><circle cx="120" cy="120" r="120" fill="url(#linear-gradient)"/><path d="M81.229,128.772l14.237,39.406s1.78,3.687,3.686,3.687,30.255-29.492,30.255-29.492l31.525-60.89L81.737,118.6Z" fill="#c8daea"/><path d="M100.106,138.878l-2.733,29.046s-1.144,8.9,7.754,0,17.415-15.763,17.415-15.763" fill="#a9c6d8"/><path d="M81.486,130.178,52.2,120.636s-3.5-1.42-2.373-4.64c.232-.664.7-1.229,2.1-2.2,6.489-4.523,120.106-45.36,120.106-45.36s3.208-1.081,5.1-.362a2.766,2.766,0,0,1,1.885,2.055,9.357,9.357,0,0,1,.254,2.585c-.009.752-.1,1.449-.169,2.542-.692,11.165-21.4,94.493-21.4,94.493s-1.239,4.876-5.678,5.043A8.13,8.13,0,0,1,146.1,172.5c-8.711-7.493-38.819-27.727-45.472-32.177a1.27,1.27,0,0,1-.546-.9c-.093-.469.417-1.05.417-1.05s52.426-46.6,53.821-51.492c.108-.379-.3-.566-.848-.4-3.482,1.281-63.844,39.4-70.506,43.607A3.21,3.21,0,0,1,81.486,130.178Z" fill="#fff"/></svg>
				</template>

                <div class="grow">
                    <div class="flex flex-col gap-1">
                        <div class="grid grid-cols-12">
                            <div class="col-span-11 flex">
                                <h4
                                    class="truncate"
                                    x-text="chatItem.conversation_name"
                                ></h4>
                            </div>
                            <div class="col-span-1 flex justify-center text-[11px] opacity-50">
                                <p class="m-0 whitespace-nowrap">
                                    <span
                                        x-text="getShortDiffHumanTime(Math.floor((new Date() - new Date(chatItem?.lastMessage?.created_at || chatItem.created_at)) / 1000))"></span>
                                </p>
                            </div>
                        </div>
                        <div class="grid grid-cols-12">
                            <div class="col-span-11 flex">
                                <p
                                    class="mb-0.5 line-clamp-[2] text-ellipsis text-xs"
                                    x-text="chatItem.lastMessage?.message ? chatItem.lastMessage?.message :  '{{ __('Chat history item') }}'"
                                    :class="{ 'font-semibold text-heading-foreground': !chatItem.lastMessage?.read_at }"
                                    :class="{ 'text-heading-foreground/30': chatItem.lastMessage?.read_at ? true : false }"
                                ></p>
                            </div>
                            <div class="col-span-1 flex justify-center">
                                <span
                                    class="ms-auto flex size-5 items-center justify-center rounded-full bg-[#F3E2FD] text-xs text-black"
                                    x-show="getUnreadMessages(chatItem.id)"
                                    x-text="getUnreadMessages(chatItem.id)"
                                ></span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <a
                class="lqd-ext-chatbot-history-list-item-trigger absolute start-0 top-0 inline-block h-full w-full"
                :data-id="chatItem.id"
                href="#"
                title="{{ __('Open Chat History') }}"
                @click.prevent="setActiveChat"
            ></a>
        </li>
    </template>
    <template x-if="!chatsList.length">
        <p class="mb-0.5 font-semibold text-heading-foreground">
            {{ __('No chat history found.') }}
        </p>
    </template>
</ul>
