@php
	use App\Helpers\Classes\MarketplaceHelper;

	$disable_actions = $app_is_demo ?? false;
	$is_search = $is_search ?? false;
	$categoryId = isset($category) ? $category->id : null;
	$website_url = $website_url ?? null;
	$isChatProImage = ($website_url ?? null) === 'chatpro-image' && MarketplaceHelper::isRegistered('ai-chat-pro-image-chat');
	$isChatProContext = in_array($website_url, ['chatpro', 'chatpro-temp', 'chatPro', 'chatpro-image', 'social-media-agent'], true);
	$chatProFoldersEnabled = $isChatProContext && MarketplaceHelper::isRegistered('ai-chat-pro-folders');
@endphp

<div
	class="chat-list-wrapper h-full"
	x-data="chatSidebarManager({
        categoryId: {{ $categoryId ?? 'null' }},
        isSearch: {{ $is_search ? 'true' : 'false' }},
        websiteUrl: '{{ $website_url ?? '' }}',
        foldersEnabled: {{ $chatProFoldersEnabled ? 'true' : 'false' }},
        isChatProImage: {{ $isChatProImage ? 'true' : 'false' }},
        disableActions: {{ $disable_actions ? 'true' : 'false' }}
    })"
>
	{{-- Initial Loading Skeleton --}}
	<div
		class="flex flex-col gap-2"
		x-show="initialLoading"
		x-cloak
	>
		<template x-for="i in 10">
			<div class="chat-list-item-trigger pointer-events-none flex animate-pulse items-center gap-2 p-5 ps-8">
				<div class="h-4 w-3/4 rounded bg-foreground/10"></div>
			</div>
		</template>
	</div>

	<ul
		class="chat-list-ul flex h-full flex-col overflow-y-auto text-xs"
		x-show="!initialLoading"
	>
		@if ($chatProFoldersEnabled)
			{{-- Back Button When A Folder Is Open --}}
			<li
				class="chat-folder-item chat-list-item group relative flex shrink-0 items-center gap-2"
				x-cloak
				x-show="openFolder"
			>
				<div
					class="chat-list-item-trigger relative flex grow cursor-pointer items-center gap-2 p-5 ps-8"
					@click="toggleFolder(null)"
				>
					<x-tabler-arrow-left
						class="h-[1lh] w-5 shrink-0 rtl:rotate-180"
						stroke-width="1.5"
					/>

					<span class="chat-item-title">
                        @lang('Back to Chats')
                    </span>
				</div>
			</li>

			{{-- Add Folder --}}
			<li
				class="chat-folder-item chat-add-folder-item chat-list-item group relative flex shrink-0 items-center gap-2 [word-break:break-word]"
				x-show="!openFolder"
			>
				<x-button
					class="chat-add-folder-modal-trigger chat-list-item-trigger w-full justify-start gap-2 p-5 ps-8 text-xs font-normal"
					variant="link"
					@click.prevent="toggleAddFolderModal(true)"
				>
					<x-tabler-folder-plus class="h-[1lh] w-5 shrink-0" />

					<span class="chat-item-title">
                        @lang('Add Folder')
                    </span>
				</x-button>
			</li>

			{{-- Folders List --}}
			<template
				x-for="folder in folders"
				:key="folder.id"
			>
				<li
					class="chat-folder-item chat-list-item group relative flex shrink-0 items-center gap-2"
					x-show="!openFolder"
					:class="{ 'active': chatsList.find(chat => chat.id === openChat && chat.folder_id === folder.id) }"
				>
					<div
						class="chat-list-item-trigger relative flex grow cursor-pointer items-center gap-2 p-5 ps-8"
						@click="toggleFolder(folder.id)"
					>
						<x-tabler-folder class="h-[1lh] w-5 shrink-0" />

						<span
							class="chat-item-title grow"
							x-text="folder.name"
						></span>

						<x-dropdown.dropdown
							class="order-last ms-auto shrink-0"
							offsetY="10px"
						>
							<x-slot:trigger
								class="inline-grid size-8 place-items-center p-0 md:size-5 md:opacity-0 md:group-hover:opacity-100 md:group-[&.lqd-is-active]/dropdown:opacity-100"
								variant="none"
							>
								<x-tabler-dots class="size-5 opacity-70" />
							</x-slot:trigger>

							<x-slot:dropdown
								class="max-h-72 min-w-[200px] overflow-y-auto px-2 py-2"
							>
								<x-button
									class="w-full justify-start rounded-md px-3.5 py-2 text-start text-2xs font-medium text-foreground/70 hover:transform-none hover:bg-heading-foreground/[3%] hover:text-heading-foreground"
									variant="none"
									@click.prevent.stop="toggle('collapse'); setSelectedItem({type: 'folder', id: folder.id, title: folder.name}); toggleRenameModal(true);"
								>
									<x-tabler-pencil class="size-5 shrink-0" />
									@lang('Rename')
								</x-button>

								<hr class="my-1 opacity-50">

								<x-button
									class="w-full justify-start rounded-md px-3.5 py-2 text-start text-2xs font-medium text-foreground/70 hover:transform-none hover:bg-heading-foreground/[3%] hover:text-heading-foreground"
									variant="none"
									@click.prevent.stop="toggle('collapse'); setSelectedItem({type: 'folder', id: folder.id, title: folder.name}); toggleDeleteModal(true);"
								>
									<x-tabler-trash class="size-5 shrink-0 text-red-500" />
									@lang('Delete Folder')
								</x-button>
							</x-slot:dropdown>
						</x-dropdown.dropdown>
					</div>
				</li>
			</template>
		@endif

		{{-- Chats List --}}
		<template
			x-for="chat in chatsListToShow"
			:key="chat.id"
		>
			<li
				class="chat-list-item group relative !order-none flex shrink-0 items-center gap-1 border-b [word-break:break-word]"
				:id="'chat_' + chat.id"
				:class="{
                    'pin-mode': chat.is_pinned,
                    'active': openChat == chat.id,
                    '[&.active]:before:absolute [&.active]:before:left-0 [&.active]:before:top-[25%] [&.active]:before:h-[50%] [&.active]:before:w-[3px] [&.active]:before:bg-gradient-to-b [&.active]:before:from-primary [&.active]:before:to-transparent': true
                }"
			>
				<div
					class="chat-list-item-trigger flex grow cursor-pointer gap-2 p-5 ps-8 text-start text-heading-foreground"
					@click="await openChatAreaContainer(chat.id, '{{ $website_url ?? '' }}'); mobileSidebarShow = false; $nextTick(() => { openChat = chat.id })"
				>
					{{-- Thumbnail for image-pro chats --}}
					<template x-if="isChatProImage && chat.thumbnail_url">
						<div class="lqd-chat-item-thumb-wrap relative size-6 shrink-0 overflow-hidden rounded bg-foreground/5">
							<img
								class="lqd-chat-item-thumb size-full object-cover"
								:src="chat.thumbnail_url"
								alt="{{ __('Generated image preview') }}"
								loading="lazy"
								onerror="this.closest('.lqd-chat-item-thumb-wrap')?.classList.add('hidden'); this.closest('.chat-list-item-trigger')?.querySelector('.lqd-chat-item-trigger-icons')?.classList.remove('hidden');"
							>
						</div>
					</template>

					<div
						class="lqd-chat-item-trigger-icons flex flex-col gap-y-2"
						:class="{ 'hidden': isChatProImage && chat.thumbnail_url }"
					>
						<x-tabler-pinned
							class="lqd-chat-item-trigger-icon-pin hidden size-6 group-[&.pin-mode]:block"
							stroke-width="1.5"
						/>
						<x-tabler-message
							class="lqd-chat-item-trigger-icon-message size-6 shrink-0 group-[&.pin-mode]:hidden"
							stroke-width="1.5"
						/>
					</div>

					<span class="lqd-chat-item-trigger-info flex grow flex-col">
                        <span
							class="chat-item-title text-xs font-medium group-[&.edit-mode]:pointer-events-auto"
							x-text="chat.title"
						></span>
                        <span
							class="chat-item-date text-3xs opacity-40"
							x-text="getFormattedDateTime(chat.updated_at)"
						></span>
                        <template x-if="chat.reference_url">
                            <a
								class="flex underline opacity-90"
								target="_blank"
								:title="chat.reference_url"
								:href="chat.reference_url"
								@click.stop
							>
                                <span x-text="chat.doc_name"></span>
                            </a>
                        </template>
                        <template x-if="chat.website_url">
                            <a
								class="flex underline opacity-90"
								target="_blank"
								:title="chat.website_url"
								:href="chat.website_url"
								@click.stop
							>
                                <span x-text="chat.website_url"></span>
                            </a>
                        </template>
                    </span>

					{{-- Chat Dropdown Menu --}}
					<x-dropdown.dropdown
						class="order-last ms-auto shrink-0"
						offsetY="10px"
					>
						<x-slot:trigger
							class="inline-grid size-8 place-items-center p-0 md:size-5 md:opacity-0 md:group-hover:opacity-100 md:group-[&.lqd-is-active]/dropdown:opacity-100"
							variant="none"
						>
							<x-tabler-dots class="size-5 opacity-70" />
						</x-slot:trigger>

						<x-slot:dropdown
							class="max-h-72 min-w-[200px] overflow-y-auto px-2 py-2"
						>
							<div
								class="grid grid-cols-1 place-items-start"
								x-data="{ activeView: 'root' }"
							>
								{{-- Root Actions --}}
								<div
									class="col-start-1 col-end-1 row-start-1 row-end-1 w-full"
									x-show="activeView === 'root'"
								>
									<x-button
										class="w-full justify-start rounded-md px-3.5 py-2 text-start text-2xs font-medium text-foreground/70 hover:transform-none hover:bg-heading-foreground/[3%] hover:text-heading-foreground"
										variant="none"
										@click.prevent.stop="toggle('collapse'); setSelectedItem({type: 'chatItem', id: chat.id, title: chat.title}); toggleRenameModal(true);"
									>
										<x-tabler-pencil class="size-5 shrink-0" />
										@lang('Rename')
									</x-button>

									<x-button
										class="w-full justify-start rounded-md px-3.5 py-2 text-start text-2xs font-medium text-foreground/70 hover:transform-none hover:bg-heading-foreground/[3%] hover:text-heading-foreground"
										variant="none"
										@click.prevent.stop="toggle('collapse'); pinChat(chat.id, !chat.is_pinned)"
									>
										<x-tabler-pin
											class="size-5 shrink-0"
											x-show="!chat.is_pinned"
										/>
										<x-tabler-pinned
											class="size-5 shrink-0"
											x-cloak
											x-show="chat.is_pinned"
										/>
										<span x-text="chat.is_pinned ? '{{ __('Unpin') }}' : '{{ __('Pin') }}'"></span>
									</x-button>

									@if ($chatProFoldersEnabled)
										<x-button
											class="w-full justify-start rounded-md px-3.5 py-2 text-start text-2xs font-medium text-foreground/70 hover:transform-none hover:bg-heading-foreground/[3%] hover:text-heading-foreground"
											variant="none"
											@click.prevent.stop="activeView = 'moveToFolder'"
										>
											<x-tabler-folder class="size-5 shrink-0" />
											@lang('Move to Folder')
											<x-tabler-chevron-right class="ms-auto size-4 rtl:rotate-180" />
										</x-button>
									@endif

									<hr class="my-1 opacity-50">

									<x-button
										class="w-full justify-start rounded-md px-3.5 py-2 text-start text-2xs font-medium text-foreground/70 hover:transform-none hover:bg-heading-foreground/[3%] hover:text-heading-foreground"
										variant="none"
										@click.prevent.stop="toggle('collapse'); setSelectedItem({type: 'chatItem', id: chat.id, title: chat.title}); toggleDeleteModal(true);"
									>
										<x-tabler-trash class="size-5 shrink-0 text-red-500" />
										@lang('Delete Chat')
									</x-button>
								</div>

								{{-- Move To Folder Sub-view --}}
								@if ($chatProFoldersEnabled)
									<div
										class="col-start-1 col-end-1 row-start-1 row-end-1 w-full"
										x-show="activeView === 'moveToFolder'"
										x-cloak
									>
										<x-button
											class="w-full justify-start rounded-md px-3.5 py-2 text-start text-2xs font-medium text-foreground/70 hover:transform-none hover:bg-heading-foreground/[3%] hover:text-heading-foreground"
											variant="none"
											@click.prevent.stop="activeView = 'root'"
										>
											<x-tabler-arrow-left class="h-[1lh] w-5 rtl:rotate-180" />
											@lang('Back')
										</x-button>

										<hr class="my-1 opacity-50">

										<x-button
											class="w-full justify-start rounded-md px-3.5 py-2 text-start text-2xs font-medium text-foreground/70 hover:transform-none hover:bg-heading-foreground/[3%] hover:text-heading-foreground"
											variant="none"
											@click.prevent.stop="moveToFolder(chat.id, null)"
											x-cloak
											x-show="chat.folder_id"
										>
											<x-tabler-folder-x class="h-[1lh] w-5 shrink-0" />
											<span
												x-text="`{{ __('Remove from') }} ${folders.find(folder => folder.id == chat.folder_id)?.name ?? '{{ __('Folder') }}'}`"></span>
										</x-button>

										<x-button
											class="w-full justify-start rounded-md px-3.5 py-2 text-start text-2xs font-medium text-foreground/70 hover:transform-none hover:bg-heading-foreground/[3%] hover:text-heading-foreground"
											variant="none"
											@click.prevent.stop="toggle('collapse'); toggleAddFolderModal(true, chat.id);"
										>
											<x-tabler-folder-plus class="h-[1lh] w-5 shrink-0" />
											@lang('Add New Folder')
										</x-button>

										<hr class="my-1 opacity-50">

										<template
											x-for="folder in folders.filter(folder => folder.id !== chat.folder_id)"
											:key="folder.id"
										>
											<x-button
												class="w-full justify-start rounded-md px-3.5 py-2 text-start text-2xs font-medium text-foreground/70 hover:transform-none hover:bg-heading-foreground/[3%] hover:text-heading-foreground"
												variant="none"
												x-text="folder.name"
												@click.prevent.stop="moveToFolder(chat.id, folder.id)"
											></x-button>
										</template>
									</div>
								@endif
							</div>
						</x-slot:dropdown>
					</x-dropdown.dropdown>
				</div>
			</li>
		</template>

		{{-- Load More / Loading Indicator --}}
		<li
			class="chat-list-item flex shrink-0 items-center justify-center py-4"
			x-show="loadingMore || fetchingChats"
			x-cloak
		>
			<x-tabler-loader-2 class="size-5 animate-spin text-foreground/50" />
		</li>

		{{-- Empty State --}}
		<li
			class="chat-list-item flex shrink-0 flex-col items-center justify-center gap-2 p-8 text-center text-foreground/70"
			x-show="!initialLoading && !fetchingChats && chatsListToShow.length === 0 && !loadingMore"
			x-cloak
		>
			<x-tabler-message-circle-off class="size-8" />
			<p
				class="text-xs font-medium"
				x-text="openFolder ? '{{ __('No chats in this folder') }}' : '{{ __('No chats yet') }}'"
			></p>
		</li>
	</ul>

	@if ($chatProFoldersEnabled)
		{{-- Add Folder Modal --}}
		<x-modal
			class="chat-add-folder-modal"
			class:modal-content="w-[min(calc(100%-2rem),600px)] lg:overflow-visible"
			class:modal-backdrop="bg-black/40"
			class:modal-head="lg:border-none lg:p-0"
			class:modal-body="px-7 py-10"
			class:close-btn="lg:absolute lg:top-0 lg:-end-12 lg:bg-background lg:rounded-full lg:size-[35px] lg:hover:bg-background lg:hover:scale-105"
			x-init="$watch('modalOpen', open => $nextTick(changeSidebarLock(open)))"
		>
			<x-slot:modal>
				<form
					class="space-y-7"
					@submit.prevent="createFolder"
				>
					<div>
						<h3 class="mb-3">
							@lang('Add Folder')
						</h3>
						<p class="mb-0 text-balance">
							@lang('Organize your chat history by creating custom folders and sorting your conversations into them. ')
						</p>
					</div>

					<x-forms.input
						class:label="font-medium text-xs"
						label="{{ __('Name') }}"
						placeholder="Work"
						size="lg"
						name="folderName"
					/>

					<x-button
						class="w-full"
						variant="secondary"
						::disabled="loading"
						type="submit"
					>
						@lang('Create Folder')
						<span class="inline-grid size-7 place-items-center rounded-full bg-background text-foreground">
                            <x-tabler-chevron-right
								class="size-4"
								x-show="!loading"
							/>
                            <x-tabler-loader-2
								class="size-4 animate-spin"
								x-cloak
								x-show="loading"
							/>
                        </span>
					</x-button>
				</form>
			</x-slot:modal>
		</x-modal>
	@endif

	{{-- Rename Modal --}}
	<x-modal
		class="chat-rename-modal"
		class:modal-content="w-[min(calc(100%-2rem),600px)] lg:overflow-visible"
		class:modal-backdrop="bg-black/40"
		class:modal-head="lg:border-none lg:p-0"
		class:modal-body="px-7 py-10"
		class:close-btn="lg:absolute lg:top-0 lg:-end-12 lg:bg-background lg:rounded-full lg:size-[35px] lg:hover:bg-background lg:hover:scale-105"
		x-init="$watch('modalOpen', open => $nextTick(changeSidebarLock(open)))"
	>
		<x-slot:modal>
			<form
				class="space-y-7"
				@submit.prevent="renameItem"
			>
				<div>
					<h3 class="mb-3">
						@lang('Renaming')
						<span
							class="underline underline-offset-4 opacity-60"
							x-text="selectedItem.type === 'folder' ? folders.find(folder => folder.id === selectedItem.id)?.name : chatsList.find(chat => chat.id === selectedItem.id)?.title"
						></span>
					</h3>
					<p
						class="m-0 text-balance"
						x-text="selectedItem.type == 'folder' ? `{{ __('Enter a new name for the folder.') }}` : `{{ __('Enter a new name for the chat.') }}`"
					></p>
				</div>

				<x-forms.input
					class:label="font-medium text-xs"
					label="{{ __('Name') }}"
					size="lg"
					name="title"
					x-model="selectedItem.title"
				/>

				<input
					type="hidden"
					name="type"
					x-model="selectedItem.type"
				>
				<input
					type="hidden"
					name="id"
					x-model="selectedItem.id"
				>

				<x-button
					class="w-full"
					variant="secondary"
					::disabled="loading"
					type="submit"
				>
					@lang('Rename')
					<span class="inline-grid size-7 place-items-center rounded-full bg-background text-foreground">
                        <x-tabler-chevron-right
							class="size-4"
							x-show="!loading"
						/>
                        <x-tabler-loader-2
							class="size-4 animate-spin"
							x-cloak
							x-show="loading"
						/>
                    </span>
				</x-button>
			</form>
		</x-slot:modal>
	</x-modal>

	{{-- Delete Modal --}}
	<x-modal
		class="chat-delete-modal"
		class:modal-content="w-[min(calc(100%-2rem),600px)] lg:overflow-visible"
		class:modal-backdrop="bg-black/40"
		class:modal-head="lg:border-none lg:p-0"
		class:modal-body="px-7 py-10"
		class:close-btn="lg:absolute lg:top-0 lg:-end-12 lg:bg-background lg:rounded-full lg:size-[35px] lg:hover:bg-background lg:hover:scale-105"
		x-init="$watch('modalOpen', open => $nextTick(changeSidebarLock(open)))"
	>
		<x-slot:modal>
			<form
				class="space-y-7"
				@submit.prevent="deleteItem"
			>
				<div>
					<h3 class="mb-3">
						@lang('Deleting')
						<span
							class="underline underline-offset-4 opacity-60"
							x-text="selectedItem.title"
						></span>
					</h3>
					@if ($chatProFoldersEnabled)
						<p
							class="m-0 text-balance"
							x-cloak
							x-show="selectedItem.type == 'folder'"
						>
							@lang('All chats inside this folder will be moved to the main chat list. This action cannot be undone.')
						</p>
					@endif
					<p
						class="m-0 text-balance"
						x-cloak
						x-show="selectedItem.type == 'chatItem'"
					>
						@lang('This chat will be permanently deleted. This action cannot be undone.')
					</p>
				</div>

				<input
					type="hidden"
					name="type"
					x-model="selectedItem.type"
				>
				<input
					type="hidden"
					name="id"
					x-model="selectedItem.id"
				>

				<div class="flex gap-2">
					<x-button
						class="w-full"
						size="lg"
						variant="outline"
						::disabled="loading"
						type="button"
						@click.prevent="setSelectedItem({type: null, id: null, title: null}); toggleDeleteModal(false);"
					>
						@lang('Cancel')
					</x-button>

					<x-button
						class="w-full"
						variant="danger"
						::disabled="loading"
						type="submit"
					>
						@lang('Delete')
						<span class="inline-grid size-7 place-items-center rounded-full bg-background text-foreground">
                            <x-tabler-trash
								class="size-4"
								x-show="!loading"
							/>
                            <x-tabler-loader-2
								class="size-4 animate-spin"
								x-cloak
								x-show="loading"
							/>
                        </span>
					</x-button>
				</div>
			</form>
		</x-slot:modal>
	</x-modal>
</div>

@pushOnce('script')
	<script>
		document.addEventListener('alpine:init', () => {
			Alpine.data('chatSidebarManager', ({
												   categoryId = null,
												   isSearch = false,
												   websiteUrl = '',
												   foldersEnabled = false,
												   isChatProImage = false,
												   disableActions = false
											   }) => ({
				chatsList: [],
				_folders: [],
				openFolder: null,
				openChat: typeof chatid !== 'undefined' ? chatid : null,
				selectedItem: {
					type: 'chatItem',
					id: null,
					title: null
				},

				loading: false,
				initialLoading: true,
				categoryId: categoryId,
				isSearch: isSearch,
				websiteUrl: websiteUrl,
				foldersEnabled: foldersEnabled,
				isChatProImage: isChatProImage,
				disableActions: disableActions,
				searchQuery: '',
				searchDebounceTimer: null,

				pagination: {
					currentPage: 1,
					lastPage: 1,
					perPage: 20,
					total: 0,
					hasMore: false
				},
				loadingMore: false,
				fetchingChats: false,

				async init() {
					if (this.foldersEnabled && !this.isSearch) {
						await this.fetchFolders();
					}

					await this.fetchChats();
					this.initialLoading = false;

					this.$nextTick(() => {
						this.setupInfiniteScroll();
						this.setupSearchListener();
						this.setupSidebarRefreshListener();
					});
				},

				setupInfiniteScroll() {
					const scrollContainer = this.$el.querySelector('.chat-list-ul');
					if (!scrollContainer) return;

					this.scrollContainer = scrollContainer;

					scrollContainer.addEventListener('scroll', () => {
						this.checkAndLoadMore();
					});

					this.$nextTick(() => {
						this.checkAndLoadMore();
					});
				},

				setupSearchListener() {
					const searchInput = document.querySelector('#chat_search_word');
					if (!searchInput) return;

					searchInput.addEventListener('input', (e) => {
						clearTimeout(this.searchDebounceTimer);
						this.searchDebounceTimer = setTimeout(() => {
							this.searchQuery = e.target.value.trim();
							this.isSearch = this.searchQuery !== '';
							this.pagination.currentPage = 1;
							this.pagination.hasMore = false;
							this.fetchChats(false);
						}, 300);
					});

					// Override global searchChatFunction so the old jQuery handler becomes a no-op
					if (typeof window.searchChatFunction === 'function') {
						window._originalSearchChatFunction = window.searchChatFunction;
						window.searchChatFunction = () => {};
					}
				},

				setupSidebarRefreshListener() {
					document.addEventListener('chat-sidebar:refresh', () => {
						this.pagination.currentPage = 1;
						this.pagination.hasMore = false;
						this.fetchChats(false);
						if (this.foldersEnabled) {
							this.fetchFolders();
						}
					});

					document.addEventListener('chat-sidebar:set-active', (e) => {
						if (e.detail?.chatId) {
							this.openChat = e.detail.chatId;
						}
					});
				},

				checkAndLoadMore() {
					if (this.loadingMore || !this.pagination.hasMore || !this.scrollContainer) return;

					const {
						scrollTop,
						scrollHeight,
						clientHeight
					} = this.scrollContainer;

					const nearBottom = scrollTop + clientHeight >= scrollHeight - 100;
					const contentDoesntFill = scrollHeight <= clientHeight;

					if (nearBottom || contentDoesntFill) {
						this.loadMore();
					}
				},

				get folders() {
					return this._folders.sort((a, b) => a.name.localeCompare(b.name));
				},

				set folders(folders) {
					this._folders = folders;
				},

				get chatsListToShow() {
					if (this.isSearch) {
						return this.chatsList.sort((a, b) => {
							if (a.is_pinned !== b.is_pinned) {
								return b.is_pinned - a.is_pinned;
							}
							return new Date(b.created_at) - new Date(a.created_at);
						});
					}

					if (this.foldersEnabled) {
						return this.chatsList
							.filter(chat => chat.folder_id == this.openFolder)
							.sort((a, b) => {
								if (a.is_pinned !== b.is_pinned) {
									return b.is_pinned - a.is_pinned;
								}
								return new Date(b.created_at) - new Date(a.created_at);
							});
					}

					return this.chatsList.sort((a, b) => {
						if (a.is_pinned !== b.is_pinned) {
							return b.is_pinned - a.is_pinned;
						}
						return new Date(b.created_at) - new Date(a.created_at);
					});
				},

				async fetchFolders() {
					try {
						const response = await this.makeRequest('/dashboard/user/ai-chat-pro/folders', {
							method: 'GET'
						});

						if (response.success) {
							this._folders = response.folders;
						}
					} catch (error) {
						console.error('Fetch folders error:', error);
					}
				},

				async fetchChats(append = false) {
					if (this.loadingMore || this.fetchingChats) return;

					const page = append ? this.pagination.currentPage + 1 : 1;

					try {
						this.fetchingChats = true;
						if (append) {
							this.loadingMore = true;
						}

						const params = new URLSearchParams({
							per_page: this.pagination.perPage,
							page: page
						});

						if (this.categoryId) {
							params.append('category_id', this.categoryId);
						}

						if (this.websiteUrl) {
							params.append('website_url', this.websiteUrl);
						}

						if (this.searchQuery) {
							params.append('search', this.searchQuery);
						}

						if (this.foldersEnabled && this.openFolder !== null && !this.isSearch) {
							params.append('folder_id', this.openFolder);
						}

						const response = await this.makeRequest(
							`/dashboard/user/openai/chat/get-chats?${params.toString()}`, {
								method: 'GET'
							}
						);

						if (response.success) {
							if (append) {
								const existingIds = new Set(this.chatsList.map(c => c.id));
								const newChats = response.chats.filter(c => !existingIds.has(c.id));
								this.chatsList = [...this.chatsList, ...newChats];
							} else {
								this.chatsList = response.chats;
							}

							this.pagination = {
								currentPage: response.pagination.current_page,
								lastPage: response.pagination.last_page,
								perPage: response.pagination.per_page,
								total: response.pagination.total,
								hasMore: response.pagination.has_more
							};
						}
					} catch (error) {
						console.error('Fetch chats error:', error);
					} finally {
						this.loadingMore = false;
						this.fetchingChats = false;
					}
				},

				async loadMore() {
					if (!this.pagination.hasMore || this.loadingMore) return;
					await this.fetchChats(true);

					this.$nextTick(() => {
						this.checkAndLoadMore();
					});
				},

				setSelectedItem({
									type,
									id,
									title
								}) {
					this.selectedItem = {
						type,
						id,
						title
					};
				},

				async toggleFolder(id) {
					if (!this.foldersEnabled) return;

					const previousFolder = this.openFolder;
					this.openFolder = id;

					if (id !== null && id !== previousFolder) {
						this.pagination.currentPage = 1;
						this.pagination.hasMore = false;
						await this.fetchChats(false);
					} else if (id === null && previousFolder !== null) {
						this.pagination.currentPage = 1;
						this.pagination.hasMore = false;
						await this.fetchChats(false);
					}
				},

				toggleAddFolderModal(open, chatId = null) {
					if (!this.foldersEnabled) return;

					const modalEl = document.querySelector('.chat-add-folder-modal');

					if (!modalEl) {
						return toastr.error('{{ __('Could not find the modal element.') }}');
					}

					if (chatId) {
						this.setSelectedItem({
							type: 'chatItem',
							id: chatId,
							title: null
						});
					} else {
						this.setSelectedItem({
							type: null,
							id: null,
							title: null
						});
					}

					Alpine.$data(modalEl).modalOpen = open != null ? open : !Alpine.$data(modalEl).modalOpen;
				},

				toggleRenameModal(open) {
					const modalEl = document.querySelector('.chat-rename-modal');

					if (!modalEl) {
						return toastr.error('{{ __('Could not find the modal element.') }}');
					}

					Alpine.$data(modalEl).modalOpen = open != null ? open : !Alpine.$data(modalEl).modalOpen;
				},

				toggleDeleteModal(open) {
					const modalEl = document.querySelector('.chat-delete-modal');

					if (!modalEl) {
						return toastr.error('{{ __('Could not find the modal element.') }}');
					}

					Alpine.$data(modalEl).modalOpen = open != null ? open : !Alpine.$data(modalEl).modalOpen;
				},

				updateChatItem(chatId, options) {
					this.chatsList = this.chatsList.map(chat => {
						if (chat.id == chatId) {
							Object.keys(options).forEach(key => {
								chat[key] = options[key];
							});
						}
						return chat;
					});
				},

				updateFolder(folderId, options) {
					this._folders = this._folders.map(folder => {
						if (folder.id == folderId) {
							Object.keys(options).forEach(key => {
								folder[key] = options[key];
							});
						}
						return folder;
					});
				},

				async createFolder(event) {
					if (!this.foldersEnabled) return;

					const formData = new FormData(event.target);
					const folderName = formData.get('folderName')?.trim() || '';

					if (!folderName) {
						toastr.error('{{ __('Please enter a folder name') }}');
						return;
					}

					if (this.folders.some(f => f.name.toLowerCase() === folderName.toLowerCase())) {
						toastr.error('{{ __('A folder with this name already exists') }}');
						return;
					}

					this.loading = true;

					try {
						const response = await this.makeRequest('/dashboard/user/ai-chat-pro/folders', {
							method: 'POST',
							body: JSON.stringify({
								name: folderName
							})
						});

						if (response.success) {
							toastr.success(response.message || '{{ __('Folder created successfully') }}');
							this._folders.push(response.folder);

							const addFolderModal = document.querySelector('.chat-add-folder-modal');

							if (addFolderModal) {
								Alpine.$data(addFolderModal).modalOpen = false;
							}

							event.target.reset();

							if (this.selectedItem.type === 'chatItem' && this.selectedItem.id) {
								await this.moveToFolder(this.selectedItem.id, response.folder.id);
							}
						}
					} catch (error) {
						console.error('Create folder error:', error);
						toastr.error(error.message || '{{ __('Failed to create folder') }}');
					} finally {
						this.loading = false;
					}
				},

				async renameItem(event) {
					const formData = new FormData(event.target);
					const id = formData.get('id');
					const type = formData.get('type');
					const title = formData.get('title');

					if (type === 'folder') {
						await this.renameFolder(id, title);
					} else if (type === 'chatItem') {
						await this.renameChatItem(id, title);
					}
				},

				async renameFolder(folderId, folderName) {
					if (!this.foldersEnabled || !folderId || !folderName) {
						return toastr.error('{{ __('Please fill all required fields.') }}');
					}

					if (this.folders.some(f =>
						f.id != folderId &&
						f.name.toLowerCase() === folderName.toLowerCase()
					)) {
						return toastr.error('{{ __('A folder with this name already exists') }}');
					}

					this.loading = true;

					try {
						const response = await this.makeRequest(
							`/dashboard/user/ai-chat-pro/folders/${folderId}`, {
								method: 'PUT',
								body: JSON.stringify({
									name: folderName
								})
							}
						);

						if (response.success) {
							toastr.success(response.message || '{{ __('Folder renamed successfully') }}');

							this.updateFolder(folderId, {
								name: response.folder.name
							});

							const renameModal = document.querySelector('.chat-rename-modal');

							if (renameModal) {
								Alpine.$data(renameModal).modalOpen = false;
							}
						}
					} catch (error) {
						console.error('Rename folder error:', error);
						toastr.error(error.message || '{{ __('Failed to rename folder') }}');
					} finally {
						this.loading = false;
					}
				},

				async renameChatItem(chatId, title) {
					if (this.disableActions) {
						return toastr.info('{{ __('This feature is disabled in Demo version.') }}');
					}

					if (!chatId || !title) {
						return toastr.error('{{ __('Please fill all required fields.') }}');
					}

					this.loading = true;

					try {
						const response = await this.makeRequest(
							'/dashboard/user/openai/chat/rename-chat', {
								method: 'POST',
								body: JSON.stringify({
									chat_id: `chat_${chatId}`,
									title
								})
							}
						);

						if (response.success) {
							toastr.success(response.message || '{{ __('Chat renamed successfully') }}');

							this.updateChatItem(chatId, {
								title
							});

							const renameModal = document.querySelector('.chat-rename-modal');

							if (renameModal) {
								Alpine.$data(renameModal).modalOpen = false;
							}
						}
					} catch (error) {
						console.error('Rename chat error:', error);
						toastr.error(error.message || '{{ __('Failed to rename chat') }}');
					} finally {
						this.loading = false;
					}
				},

				async deleteItem(event) {
					const formData = new FormData(event.target);
					const id = formData.get('id');
					const type = formData.get('type');

					if (type === 'folder') {
						await this.deleteFolder(id);
					} else if (type === 'chatItem') {
						await this.deleteChatItem(id);
					}
				},

				async deleteFolder(folderId) {
					if (!this.foldersEnabled || !folderId) {
						return toastr.error('{{ __('Please fill all required fields.') }}');
					}

					this.loading = true;

					try {
						const response = await this.makeRequest(
							`/dashboard/user/ai-chat-pro/folders/${folderId}`, {
								method: 'DELETE'
							}
						);

						if (response.success) {
							toastr.success(response.message || '{{ __('Folder deleted successfully') }}');

							this.chatsList = this.chatsList.map(chat => {
								if (chat.folder_id == folderId) {
									return {
										...chat,
										folder_id: null
									};
								}
								return chat;
							});

							this._folders = this._folders.filter(folder => folder.id != folderId);

							this.openFolder = null;
							await this.fetchChats(false);

							const deleteModal = document.querySelector('.chat-delete-modal');

							if (deleteModal) {
								Alpine.$data(deleteModal).modalOpen = false;
							}
						}
					} catch (error) {
						console.error('Delete folder error:', error);
						toastr.error(error.message || '{{ __('Failed to delete folder') }}');
					} finally {
						this.loading = false;
					}
				},

				async deleteChatItem(chatId) {
					if (this.disableActions) {
						return toastr.info('{{ __('This feature is disabled in Demo version.') }}');
					}

					if (!chatId) {
						return toastr.error('{{ __('Please fill all required fields.') }}');
					}

					this.loading = true;

					try {
						const response = await this.makeRequest(
							'/dashboard/user/openai/chat/delete-chat', {
								method: 'POST',
								body: JSON.stringify({
									chat_id: `chat_${chatId}`
								})
							}
						);

						if (response.success) {
							toastr.success(response.message || '{{ __('Chat deleted successfully') }}');

							this.chatsList = this.chatsList.filter(chat => chat.id != chatId);

							const firstChatItem = this.chatsListToShow[0];

							if (firstChatItem) {
								const chatItemEl = document.querySelector(`#chat_${firstChatItem.id}`);

								if (chatItemEl) {
									chatItemEl.querySelector('.chat-list-item-trigger').click();
								}
							}

							const deleteModal = document.querySelector('.chat-delete-modal');

							if (deleteModal) {
								Alpine.$data(deleteModal).modalOpen = false;
							}
						}
					} catch (error) {
						console.error('Delete chat error:', error);
						toastr.error(error.message || '{{ __('Failed to delete chat') }}');
					} finally {
						this.loading = false;
					}
				},

				async moveToFolder(chatId, folderId) {
					if (!this.foldersEnabled) return;

					try {
						const response = await this.makeRequest(
							`/dashboard/user/ai-chat-pro/chats/${chatId}/move-to-folder`, {
								method: 'POST',
								body: JSON.stringify({
									folder_id: folderId
								})
							}
						);

						if (response.success) {
							toastr.success(response.message || '{{ __('Chat moved successfully') }}');

							this.updateChatItem(chatId, {
								folder_id: folderId
							});

							if (folderId != this.openFolder) {
								this.chatsList = this.chatsList.filter(chat => chat.id != chatId);
							}
						}
					} catch (error) {
						console.error('Move chat error:', error);
						toastr.error(error.message || '{{ __('Failed to move chat') }}');
					}
				},

				async pinChat(chatId, pinned) {
					if (this.disableActions) {
						return toastr.info('{{ __('This feature is disabled in Demo version.') }}');
					}

					if (!chatId) {
						return toastr.error('{{ __('Please fill all required fields.') }}');
					}

					this.loading = true;

					try {
						const response = await this.makeRequest('/dashboard/user/openai/chat/pin-conversation', {
							method: 'POST',
							body: JSON.stringify({
								chat_id: `chat_${chatId}`,
								pinned
							})
						});

						if (response.success) {
							toastr.success(response.message || '{{ __('Chat pin changed successfully.') }}');

							this.updateChatItem(chatId, {
								is_pinned: pinned
							});
						}
					} catch (error) {
						console.error('Pin chat error:', error);
						toastr.error(error.message || '{{ __('Failed to pin chat') }}');
					} finally {
						this.loading = false;
					}
				},

				async makeRequest(url, options = {}) {
					const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content;

					if (!csrfToken) {
						throw new Error('CSRF token not found');
					}

					const response = await fetch(url, {
						...options,
						headers: {
							'Content-Type': 'application/json',
							'Accept': 'application/json',
							'X-CSRF-TOKEN': csrfToken,
							...options.headers,
						}
					});

					const data = await response.json();

					if (!response.ok) {
						throw new Error(data.message || 'Request failed');
					}

					return data;
				},

				changeSidebarLock(lock = false) {
					const chatsV2Container = document.querySelector('.lqd-chat-v2-container');

					if (!chatsV2Container) return;

					Alpine.$data(chatsV2Container).sidebarLocked = lock;
				},

				getFormattedDateTime(date) {
					const d = new Date(date);
					return d.toLocaleDateString(undefined, {
						year: 'numeric',
						month: 'short',
						day: 'numeric',
						hour: '2-digit',
						minute: '2-digit',
						hour12: false
					});
				}
			}));
		});
	</script>
@endPushOnce
