<x-card class="mt-4">
	<x-table
		class="text-xs"
		variant="plain"
	>
		<x-slot:head>
			<tr>
				<th class="ps-6">
					{{ __('Channel') }}
				</th>
				<th>
					{{ __('Channel Id') }}
				</th>
				<th>
					{{ __('Action') }}
				</th>
			</tr>
		</x-slot:head>

		<x-slot:body>

			<template x-show="channelFetch" x-for="channel in chatbotChannels" :key="channel.id">
				<tr>
					<td class="ps-6" x-text="channel.channel"></td>
					<td class="ps-6" x-text="channel.channel_id"></td>
					<td class="ps-6 flex justify-between">

						<x-modal title="{{ __('Channel Detail') }}">
							<x-slot:trigger
								class="text-2xs font-semibold text-primary"
								variant="link"
							>
								{{ __('Detail') }}
								<x-tabler-details class="size-4" />
							</x-slot:trigger>

							<x-slot:modal>
								<form
									class="flex flex-col gap-6"
								>
									<label>@lang('Webhook')</label>

									<p class="border rounded-lg p-3" x-text="channel.webhook">
									</p>

									<div class="mt-4 border-t pt-3 flex justify-between">
										<div>
											<x-button
												type="button"
												variant="primary"
												@click.prevent="copyToClipboard(channel.webhook)"
											>
												{{ __('Copy') }}
											</x-button>
											<x-button
												type="button"
												variant="danger"
												@click.prevent="deleteToClipboard(channel.id)"
											>
												@lang('Delete')
											</x-button>
										</div>


										<x-button
											@click.prevent="modalOpen = false"
											variant="outline"
										>
											{{ __('Cancel') }}
										</x-button>
									</div>
								</form>
							</x-slot:modal>
						</x-modal>
					</td>
				</tr>
			</template>

			<template x-show="! channelFetch">
				<tr>
					<td colspan="3" class="text-center">
						{{ __('No channels found') }}
					</td>
				</tr>
			</template>
		</x-slot:body>
	</x-table>
</x-card>
