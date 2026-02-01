{{-- Editing Step 3 - Train --}}
<div
	class="col-start-1 col-end-1 row-start-1 row-end-1 transition-all"
	data-step="3"
	x-data="externalChatbotChannel"
	x-show="editingStep === 5"
	x-transition:enter-start="opacity-0 -translate-x-3"
	x-transition:enter-end="opacity-100 translate-x-0"
	x-transition:leave-start="opacity-100 translate-x-0"
	x-transition:leave-end="opacity-0 translate-x-3"
>
	<h2 class="mb-3.5">
		@lang('Chatbot Channel')
	</h2>
	<p class="mb-3 text-xs/5 opacity-60 lg:max-w-[360px]">
		@lang('This step is optional, but you can enhance your chatbot experience through different channels.')
	</p>

	<div class="mt-10 lqd-social-media-cards-grid gap-3 flex ">
		@includeIf('telegram-channel::channel-card')
		@includeIf('whatsapp-channel::channel-card')
	</div>

	@include('chatbot::partials.channel-table')
</div>

@push('script')
	<script>

		(() => {
			document.addEventListener('alpine:init', () => {
				Alpine.data('externalChatbotChannel', () => ({
					activeTab: 'channel',
					fetching: false,
					storeChannelFetch: false,
					channelFetch: false,
					chatbotChannels: [],
					init() {
						this.$watch('editingStep', currentStep => {
							if (currentStep === 5) {
								this.fetchChannels();
							}
						})
					},
					async deleteToClipboard(id) {
						@if($app_is_demo)
							return toastr.error('{{ trans('This feature is disabled in demo mode.') }}')
						@endif
						if(confirm('If you delete the channel, all conversations linked to it will stop receiving messages. This may lead to some issues.')) {
							this.storeChannelFetch = true;

							const formData = new FormData();
							formData.append('channel_id', id)

							fetch('{{ route('dashboard.chatbot-multi-channel.delete') }}' , {
								method: 'POST',
								headers: {
									'Accept': 'application/json',
									'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').getAttribute('content'),
								},
								body: formData
							}).then(async response => {
								if (!response.ok) {
									const error = await response.json();
									toastr.error(error.message || 'Bir hata oluştu');
									this.storeChannelFetch = false;
									return;
								}

								const data = await response.json();

								if (data.status === 'success') {

									this.chatbotChannels = this.chatbotChannels.filter(channel => channel.id !== id);

									toastr.success(data.message);

									this.storeChannelFetch = false;
								}
								else {
									toastr.error(data.message || 'Bir hata oluştu');
									this.storeChannelFetch = false;
								}
							}).catch(error => {
								this.storeChannelFetch = false;
								toastr.error(error.message || 'İstek gönderilirken hata oluştu');
							});
						}
					},
					copyToClipboard(text) {
						navigator.clipboard.writeText(text)
							.then(() => {
								toastr.success('Webhook panoya kopyalandı');
							})
							.catch(err => {
								console.error('Kopyalama hatası:', err);
								toastr.error('Webhook kopyalanamadı');
							});
					},
					setActiveTab(tab) {
						if (tab === this.activeTab) return;

						this.activeTab = tab;
					},
					async fetchChannels() {
						this.channelFetch = true;

						const formData = new FormData();

						formData.append('chatbot_id', this.activeChatbot.id)

						fetch('{{ route('dashboard.chatbot-multi-channel.index') }}', {
							method: 'POST',
							headers: {
								'Accept': 'application/json',
								'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').getAttribute('content'),
							},
							body: formData
						}).then(async response => {
							if (!response.ok) {
								const error = await response.json();
								toastr.error(error.message || '{{ trans('An error occurred') }}');
								this.channelFetch = false;
								return;
							}

							const data = await response.json();

							if (data.status === 'success') {

								this.chatbotChannels = data.data;

								this.channelFetch = false;
							}
						})
							.catch(error => {
								this.channelFetch = false;
								toastr.error(error.message || '{{ trans('An error occurred while sending the request.') }}');
							});
					},
					storeChannel(id) {
						this.storeChannelFetch = true;

						const form = document.getElementById(id);
						const formData = new FormData(form);
						formData.append('chatbot_id', this.activeChatbot.id)

						fetch(form.action, {
							method: 'POST',
							headers: {
								'Accept': 'application/json',
								'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').getAttribute('content'),
							},
							body: formData
						}).then(async response => {

							if (!response.ok) {
								const error = await response.json();

								toastr.error(error.message || '{{ trans('An error occurred') }}');

								this.storeChannelFetch = false;

								return;
							}

							const data = await response.json();

							if (data.status === 'success') {

								this.chatbotChannels.push(data.data);

								toastr.success(data.message);

								this.storeChannelFetch = false;
							}
							else {
								toastr.error(data.message || '{{ trans('An error occurred') }}');
								this.storeChannelFetch = false;
							}
						}).catch(error => {
							this.storeChannelFetch = false;
							toastr.error(error.message || '{{ trans('An error occurred while sending the request.') }}');
						});
					}

				}));
			});
		})();
	</script>
@endpush
