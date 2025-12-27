import './bootstrap';
import { Alpine, Livewire } from '~vendor/livewire/livewire/dist/livewire.esm';
import ajax from '~nodeModules/@imacrayon/alpine-ajax';
import sort from '~nodeModules/@alpinejs/sort';
import intersect from '~nodeModules/@alpinejs/intersect';
import { fetchEventSource } from '@microsoft/fetch-event-source';
import { Sortable, MultiDrag } from 'sortablejs';
import modal from './components/modal';
import clipboard from './components/clipboard';
import assignViewCredits from './components/assignViewCredits';
import openaiRealtime from './components/realtime-frontend/openaiRealtime';
import advancedImageEditor from './components/advancedImageEditor';
import { debounce, difference, throttle, uniq } from 'lodash';
import creativeSuite from './components/creative-suite/creativeSuite';
import { lqdCustomizer, lqdCustomizerFontPicker } from './components/customizer';
import elevenlabsRealtime from './components/realtime-frontend/elevenlabsRealtime';
import tiptapEditor from './tiptapEditor';

window.fetchEventSource = fetchEventSource;
const darkMode = localStorage.getItem( 'lqdDarkMode' );
const docsViewMode = localStorage.getItem( 'docsViewMode' );
const socialMediaPostsViewMode = localStorage.getItem( 'socialMediaPostsViewMode' );
const navbarShrink = localStorage.getItem( 'lqdNavbarShrinked' );
const currentTheme = document.querySelector( 'body' ).getAttribute( 'data-theme' );
const lqdFocusModeEnabled = localStorage.getItem( currentTheme + ':lqdFocusModeEnabled' );

window.collectCreditsToFormData = function ( formData ) {
	const inputs = document.querySelectorAll( 'input[name^="entities"]' );
	inputs.forEach( input => {
		const name = input.name; // Get the input name
		const value = input.type === 'checkbox' || input.type === 'radio' ? input.checked : input.value; // Get value or checked status
		formData.append( name, value ); // Append to the formData object
	} );
};

window.Alpine = Alpine;

Alpine.plugin( ajax );
Alpine.plugin( sort );
Alpine.plugin( intersect );

Sortable.mount( new MultiDrag() );

document.addEventListener( 'alpine:init', () => {
	const persist = Alpine.$persist;

	Alpine.data( 'modal', data => modal( data ) );
	Alpine.data( 'clipboard', data => clipboard( data ) );
	Alpine.data( 'assignViewCredits', data => assignViewCredits( data ) );

	// Navbar shrink
	Alpine.store( 'navbarShrink', {
		active: persist( !!navbarShrink ).as( 'lqdNavbarShrinked' ),
		toggle( state ) {
			this.active = state ? ( state === 'shrink' ? true : false ) : !this.active;
			document.body.classList.toggle( 'navbar-shrinked', this.active );
		}
	} );

	// Navbar item
	Alpine.data( 'navbarItem', () => ( {
		dropdownOpen: false,
		toggleDropdownOpen( state ) {
			this.dropdownOpen = state ? ( state === 'collapse' ? true : false ) : !this.dropdownOpen;
		},
		item: {
			[ 'x-ref' ]: 'item',
			[ '@mouseenter' ]() {
				if ( !Alpine.store( 'navbarShrink' ).active ) return;
				const rect = this.$el.getBoundingClientRect();
				const dropdown = this.$refs.item.querySelector( '.lqd-navbar-dropdown' );
				[ 'y', 'height', 'bottom' ].forEach( prop => this.$refs.item.style.setProperty( `--item-${ prop }`, `${ rect[ prop ] }px` ) );

				if ( dropdown ) {
					const dropdownRect = dropdown.getBoundingClientRect();
					[ 'height' ].forEach( prop => this.$refs.item.style.setProperty( `--dropdown-${ prop }`, `${ dropdownRect[ prop ] }px` ) );
				}
			},
		}
	} ) );

	Alpine.data( 'navbarLink', ( { isDemo = false } ) => ( {
		isDemo: isDemo,
		isActive: false,
		dropdown: null,
		dropdownItems: [],
		dropdownLinks: [],
		init() {
			const navbar = document.querySelector('.lqd-navbar');
			const navbarInner = navbar?.querySelector( '.lqd-navbar-inner' );

			this.dropdown = this.$el.nextElementSibling && this.$el.nextElementSibling.classList.contains( 'lqd-navbar-dropdown' ) && this.$el.nextElementSibling;
			this.dropdownItems = this.dropdown ? this.dropdown.querySelectorAll( '.lqd-navbar-dropdown-link' ) : [];
			this.dropdownItems.forEach( item => this.dropdownLinks.push( item.href ) );

			this.isActive = this.$el.href === window.location.href || this.dropdownLinks.includes( window.location.href );

			this.$el.classList.toggle( 'active', this.isActive );

			this.$nextTick( () => {
				this.dropdown?.classList?.toggle( 'hidden', !this.isActive );

				if ( !navbar?.hasAttribute('data-disable-autoscroll') && navbarInner && !this.isDemo && this.isActive && this.$el.parentElement.offsetTop + this.$el.parentElement.offsetHeight > window.innerHeight ) {
					navbarInner.scrollTo( { top: this.$el.parentElement.offsetTop - ( window.innerHeight / 2 ) } );
				}
			} );
		},
	} ) );

	// Mobile nav
	Alpine.store( 'mobileNav', {
		navCollapse: true,
		toggleNav( state ) {
			this.navCollapse = state ? ( state === 'collapse' ? true : false ) : !this.navCollapse;
		},
		templatesCollapse: true,
		toggleTemplates( state ) {
			this.templatesCollapse = state ? ( state === 'collapse' ? true : false ) : !this.templatesCollapse;
		},
		searchCollapse: true,
		toggleSearch( state ) {
			this.searchCollapse = state ? ( state === 'collapse' ? true : false ) : !this.searchCollapse;
		},
	} );

	// light/dark mode
	Alpine.store( 'darkMode', {
		on: persist( !!darkMode ).as( 'lqdDarkMode' ),
		toggle() {
			this.on = !this.on;
			document.body.classList.toggle( 'theme-dark', this.on );
			document.body.classList.toggle( 'theme-light', !this.on );
		}
	} );

	// App loading indicator
	Alpine.store( 'appLoadingIndicator', {
		showing: false,
		show() {
			this.showing = true;
		},
		hide() {
			this.showing = false;
		},
		toggle() {
			this.showing = !this.showing;
		},
	} );

	Alpine.data('liquidHeaderSearch', () => ({
		modalOpen: false,
		searchTerm: '',
		isSearching: false,
		doneSearching: false,
		pending: false,
		inputFocused: false,
		timer: null,
		searchResults: '',
		recentSearchKeys: [],
		recentLunchedDocs: '',
		shortcutKey: navigator.userAgent.indexOf('Mac OS X') != -1 ? 'cmd' : 'ctrl',

		init() {
			this.applyRecentSearch = this.applyRecentSearch.bind(this);

			// Clear session storage on page load
			sessionStorage.removeItem('headear-recent-lunch');
			sessionStorage.removeItem('headear-recent-search-keys');

			// Fetch initial data
			this.fetchRecentSearchKeys();
			this.fetchRecentLunchedDocs();

			// Add global keyboard shortcuts
			this.addKeyboardShortcuts();
		},

		toggleModal(status) {
			if (status != null) {
				return this.modalOpen = status;
			}
			this.modalOpen = !this.modalOpen;
		},

		handleFocus() {
			this.inputFocused = true;
			if (!this.onlySpaces(this.searchTerm)) {
				this.doneSearching = true;
				this.pending = false;
			} else {
				this.pending = true;
				this.doneSearching = false;
			}
		},

		handleBlur() {
			this.inputFocused = false;
		},

		handleSearch() {
			if (this.onlySpaces(this.searchTerm)) {
				clearTimeout(this.timer);
				this.isSearching = false;
				this.doneSearching = false;
				this.pending = true;
			} else {
				this.isSearching = true;
				this.pending = false;
				this.doneSearching = false;

				clearTimeout(this.timer);
				this.timer = setTimeout(() => this.performSearch(), 1000);
			}
		},

		addKeyboardShortcuts() {
			window.addEventListener('keydown', e => {
				if ((e.ctrlKey || e.shiftKey || e.altKey || e.metaKey) && e.key === 'k') {
					e.preventDefault();
					e.stopPropagation();
					if (this.inputFocused) return;

					// Focus the search input
					this.$el.querySelector('.header-search-input').focus();
					this.inputFocused = true;

					if (!this.onlySpaces(this.searchTerm)) {
						this.doneSearching = true;
					}
				}
				if (e.key === 'Escape') {
					if (!this.inputFocused) return;
					this.$el.querySelector('.header-search-input').blur();
					this.inputFocused = false;
					this.doneSearching = false;
				}
			});
		},

		onlySpaces(str) {
			return str.trim().length === 0 || str === '';
		},

		async performSearch() {
			const formData = new FormData();
			formData.append('_token', document.querySelector('input[name=_token]')?.value);
			formData.append('search', this.searchTerm);

			try {
				const response = await fetch('/dashboard/api/search', {
					method: 'POST',
					body: formData
				});
				const result = await response.json();

				this.searchResults = result.html;
				this.doneSearching = true;
				this.pending = false;
				this.isSearching = false;

				// Store and update recent search keys
				sessionStorage.setItem('headear-recent-search-keys', JSON.stringify(result.keywords));
				this.recentSearchKeys = result.keywords;
			} catch (error) {
				console.error('Search error:', error);
				this.isSearching = false;
			}
		},

		async fetchRecentSearchKeys() {
			if (sessionStorage.getItem('headear-recent-search-keys') === null) {
				try {
					const response = await fetch('/dashboard/api/search/recent-search-keys');
					const result = await response.json();
					sessionStorage.setItem('headear-recent-search-keys', JSON.stringify(result.keys));
					this.recentSearchKeys = result.keys;
				} catch (error) {
					console.error('Error fetching recent search keys:', error);
				}
			} else {
				this.recentSearchKeys = JSON.parse(sessionStorage.getItem('headear-recent-search-keys'));
			}
		},

		async fetchRecentLunchedDocs() {
			if (sessionStorage.getItem('headear-recent-lunch') === null) {
				try {
					const response = await fetch('/dashboard/api/search/recent-lunch');
					const result = await response.json();
					sessionStorage.setItem('headear-recent-lunch', result.html);
					this.recentLunchedDocs = result.html;
				} catch (error) {
					console.error('Error fetching recent launched docs:', error);
				}
			} else {
				this.recentLunchedDocs = sessionStorage.getItem('headear-recent-lunch');
			}
		},

		applyRecentSearch(key) {
			this.searchTerm = key.keyword || key;
			this.$el.querySelector('.header-search-input').focus();
			this.handleSearch();
		},

		async deleteRecentSearchKey(key) {
			const keyValue = key.keyword || key;
			this.recentSearchKeys = this.recentSearchKeys.filter(searchKey =>
				(searchKey.keyword || searchKey) !== keyValue
			);

			sessionStorage.setItem('headear-recent-search-keys', JSON.stringify(this.recentSearchKeys));

			try {
				await fetch(`/dashboard/api/search/delete-search-key/${encodeURIComponent(keyValue)}`, {
					method: 'DELETE'
				});
			} catch (error) {
				console.error('Error deleting search key:', error);
			}
		}
	}));

	// Documents view mode
	Alpine.store( 'docsViewMode', {
		docsViewMode: persist( docsViewMode || 'list' ).as( 'docsViewMode' ),
		change( mode ) {
			this.docsViewMode = mode;
		}
	} );

	// Generators filter
	Alpine.store( 'generatorsFilter', {
		init() {
			const urlParams = new URLSearchParams( window.location.search );
			this.filter = urlParams.get( 'filter' ) || 'all';
		},
		filter: 'all',
		changeFilter( filter ) {
			if ( this.filter === filter ) return;
			if ( !document.startViewTransition ) {
				return this.filter = filter;
			}
			document.startViewTransition( () => this.filter = filter );
		}
	} );

	// Generator Item
	Alpine.data( 'generatorItem', () => ( {
		get isHidden() {
			return this.$store.generatorsFilter.filter !== 'all' &&
				this.$el.getAttribute( 'data-filter' ).search( this.$store.generatorsFilter.filter ) < 0;
		},
		updateDataFilter( id, isFavorite ) {
			const dataFilter = this.$el.getAttribute( 'data-filter' );
			const filterArray = new Set( dataFilter.split( ',' ) );

			if ( isFavorite ) {
				filterArray.add( 'favorite' );
			} else {
				filterArray.delete( 'favorite' );
			}

			this.$el.setAttribute( 'data-filter', Array.from( filterArray ).join( ',' ) );
		}
	} ) );

	// Documents filter
	Alpine.store( 'documentsFilter', {
		init() {
			const urlParams = new URLSearchParams( window.location.search );
			this.sort = urlParams.get( 'sort' ) || 'created_at';
			this.sortAscDesc = urlParams.get( 'sortAscDesc' ) || 'desc';
			this.filter = urlParams.get( 'filter' ) || 'all';
			this.page = urlParams.get( 'page' ) || '1';
		},
		sort: 'created_at',
		sortAscDesc: 'desc',
		filter: 'all',
		page: '1',
		changeSort( sort ) {
			if ( sort === this.sort ) {
				this.sortAscDesc = this.sortAscDesc === 'desc' ? 'asc' : 'desc';
			} else {
				this.sortAscDesc = 'desc';
			}
			this.sort = sort;
		},
		changeAscDesc( ascDesc ) {
			if ( this.ascDesc === ascDesc ) return;
			this.ascDesc = ascDesc;
		},
		changeFilter( filter ) {
			if ( this.filter === filter ) return;
			this.filter = filter;
		},
		changePage( page ) {
			if ( page === '>' || page === '<' ) {
				page = page === '>' ? Number( this.page ) + 1 : Number( this.page ) - 1;
			}

			if ( this.page === page ) return;

			this.page = page;
		},
	} );

	// Documents Selection
	Alpine.store( 'documentsSelection', {
		selectedItems: [],
		/**
		 *
		 * @param {{idArray: []; checkboxEl: HTMLInputElement; action: 'auto' | 'add' | 'remove'}} param0
		 */
		updateSelectedItems( { idArray = [], checkboxEl, action = 'auto' } ) {
			if (checkboxEl) {
				const id = checkboxEl.getAttribute('data-id');

				if (id === 'doc-select-all-visible') {
					// Handle select all visible checkbox
					this.handleSelectAllVisible(checkboxEl);
				} else {
					// Handle individual document checkbox
					this.handleIndividualSelection(checkboxEl, id);
				}
			} else if (idArray.length > 0) {
				// Handle programmatic selection with ID array
				this.handleArraySelection(idArray, action);
			}

			// Update select all checkbox state
			this.updateSelectAllCheckboxState();
		},

		handleSelectAllVisible(selectAllCheckbox) {
			const visibleCheckboxes = document.querySelectorAll('.document-checkbox:not([data-id="doc-select-all-visible"])');
			const shouldSelectAll = selectAllCheckbox.checked;

			visibleCheckboxes.forEach(checkbox => {
				const id = checkbox.getAttribute('data-id');
				checkbox.checked = shouldSelectAll;

				if (shouldSelectAll) {
					if (!this.selectedItems.includes(id)) {
						this.selectedItems.push(id);
					}
				} else {
					this.selectedItems = this.selectedItems.filter(item => item !== id);
				}
			});
		},

		handleIndividualSelection(checkbox, id) {
			if (checkbox.checked) {
				if (!this.selectedItems.includes(id)) {
					this.selectedItems.push(id);
				}
			} else {
				this.selectedItems = this.selectedItems.filter(item => item !== id);
			}
		},

		handleArraySelection(idArray, action) {
			if (action === 'add') {
				idArray.forEach(id => {
					if (!this.selectedItems.includes(id)) {
						this.selectedItems.push(id);
					}
				});
			} else if (action === 'remove') {
				this.selectedItems = this.selectedItems.filter(item => !idArray.includes(item));
			}
		},

		updateSelectAllCheckboxState() {
			const selectAllCheckbox = document.querySelector('.document-checkbox-all-visible');
			if (!selectAllCheckbox) return;

			const visibleCheckboxes = document.querySelectorAll('.document-checkbox:not([data-id="doc-select-all-visible"])');
			const checkedCheckboxes = document.querySelectorAll('.document-checkbox:not([data-id="doc-select-all-visible"]):checked');

			if (checkedCheckboxes.length === 0) {
				// None selected
				selectAllCheckbox.checked = false;
				selectAllCheckbox.classList.remove('partial');
			} else if (checkedCheckboxes.length === visibleCheckboxes.length) {
				// All selected
				selectAllCheckbox.checked = true;
				selectAllCheckbox.classList.remove('partial');
			} else {
				// Partial selection
				selectAllCheckbox.checked = false;
				selectAllCheckbox.classList.add('partial');
			}
		},

		clearSelection() {
			this.selectedItems = [];
			document.querySelectorAll('.document-checkbox').forEach(checkbox => {
				checkbox.checked = false;
			});
			this.updateSelectAllCheckboxState();
		},

		getSelectedCount() {
			return this.selectedItems.length;
		},

		isSelected(id) {
			return this.selectedItems.includes(id);
		},

		async bulkDelete(mode = 'selected', options = {}) {
			let payload = {};

			const {
				confirmAllMessage = 'Are you sure you want to delete all documents?',
				confirmSelectedMessage = 'Are you sure you want to delete the selected documents?',
				noSelectionMessage = 'Please select documents to delete.',
				deleteUrl = '/dashboard/user/openai/documents/bulk-delete',
				onSuccess = null,
				onError = null
			} = options;

			if (mode === 'all') {
				if (!confirm(confirmAllMessage)) {
					return;
				}
				payload.all = true;
			} else {
				if (this.selectedItems.length === 0) {
					alert(noSelectionMessage);
					return;
				}

				if (!confirm(confirmSelectedMessage)) {
					return;
				}

				payload.ids = this.selectedItems;
			}

			try {
				const response = await fetch(deleteUrl, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
					},
					body: JSON.stringify(payload)
				});

				const data = await response.json();

				if (data.success) {
					if (typeof toastr !== 'undefined') {
						toastr.success(data.message);
					}
					this.clearSelection();

					if (onSuccess) {
						onSuccess(data);
					} else {
						location.reload();
					}
				} else if (data.message) {
					if (typeof toastr !== 'undefined') {
						toastr.error(data.message);
					}
					if (onError) {
						onError(data);
					}
				}
			} catch (error) {
				console.error(error);
				const errorMessage = 'An error occurred while processing your request.';
				if (typeof toastr !== 'undefined') {
					toastr.error(errorMessage);
				}
				if (onError) {
					onError({ message: errorMessage, error });
				}
			}
		}
	});

	// Social media posts view mode
	Alpine.store( 'socialMediaPostsViewMode', {
		socialMediaPostsViewMode: persist( socialMediaPostsViewMode || 'list' ).as( 'socialMediaPostsViewMode' ),
		change( mode ) {
			this.socialMediaPostsViewMode = mode;
		}
	} );

	// Social media posts filter
	Alpine.store( 'socialMediaPostsFilter', {
		init() {
			const urlParams = new URLSearchParams( window.location.search );
			this.sort = urlParams.get( 'sort' ) || 'created_at';
			this.sortAscDesc = urlParams.get( 'sortAscDesc' ) || 'desc';
			this.filter = urlParams.get( 'filter' ) || 'all';
			this.page = urlParams.get( 'page' ) || '1';
		},
		sort: 'created_at',
		sortAscDesc: 'desc',
		filter: 'all',
		page: '1',
		changeSort( sort ) {
			if ( sort === this.sort ) {
				this.sortAscDesc = this.sortAscDesc === 'desc' ? 'asc' : 'desc';
			} else {
				this.sortAscDesc = 'desc';
			}
			this.sort = sort;
		},
		changeAscDesc( ascDesc ) {
			if ( this.ascDesc === ascDesc ) return;
			this.ascDesc = ascDesc;
		},
		changeFilter( filter ) {
			if ( this.filter === filter ) return;
			this.filter = filter;
		},
		changePage( page ) {
			if ( page === '>' || page === '<' ) {
				page = page === '>' ? Number( this.page ) + 1 : Number( this.page ) - 1;
			}

			if ( this.page === page ) return;

			this.page = page;
		},
	} );

	// Chats filter
	Alpine.store( 'chatsFilter', {
		init() {
			const urlParams = new URLSearchParams( window.location.search );
			this.filter = urlParams.get( 'filter' ) || 'all';
			this.setSearchStr( urlParams.get( 'search' ) || '' );
		},
		searchStr: '',
		setSearchStr( str ) {
			this.searchStr = str.trim().toLowerCase();
		},
		filter: 'all',
		changeFilter( filter ) {
			if ( this.filter === filter ) return;
			if ( !document.startViewTransition ) {
				return this.filter = filter;
			}
			document.startViewTransition( () => this.filter = filter );
		}
	} );

	// Generator V2
	Alpine.data( 'generatorV2', () => ( {
		itemsSearchStr: '',
		sideNavCollapsed: false,
		generatorStep: 0,
		selectedGenerator: null,

		setItemsSearchStr( str ) {
			this.itemsSearchStr = str.trim().toLowerCase();
			if ( this.itemsSearchStr !== '' ) {
				this.$el.closest( '.lqd-generator-sidebar' ).classList.add( 'lqd-showing-search-results' );
			} else {
				this.$el.closest( '.lqd-generator-sidebar' ).classList.remove( 'lqd-showing-search-results' );
			}
		},
		/**
		*
		* @param {'collapse' | 'expand'} state
		*/
		toggleSideNavCollapse( state ) {
			this.sideNavCollapsed = state ? ( state === 'collapse' ? true : false ) : !this.sideNavCollapsed;

			if ( this.sideNavCollapsed ) {
				if ( typeof tinymce !== 'undefined' && tinymce?.activeEditor ) {
					tinymce?.activeEditor?.focus();
				} else {
					window.editorJS?.focus();
				}
			}
		},
		setGeneratorStep( step ) {
			if ( step === this.generatorStep ) return;
			if ( !document.startViewTransition ) {
				return this.generatorStep = Number( step );
			}
			document.startViewTransition( () => this.generatorStep = Number( step ) );
		},
	} ) );

	// Chat
	Alpine.store( 'mobileChat', {
		sidebarOpen: false,
		toggleSidebar( state ) {
			this.sidebarOpen = state ? ( state === 'collapse' ? false : false ) : !this.sidebarOpen;
		}
	} );

	// Dropdown
	Alpine.data( 'dropdown', ( { triggerType = 'hover', preferredAnchor = 'start', offsetY = '0px', teleport = true } ) => ( {
		open: false,
		triggerType: triggerType || 'hover',
		preferredAnchor: preferredAnchor || 'start',
		offsetY: offsetY.trim() || '0px',
		teleport: teleport ?? true,
		parentRect: {
			top: 0,
			left: 0,
			width: 0,
			height: 0
		},
		dropdownRect: {
			top: 0,
			left: 0,
			width: 0,
			height: 0
		},
		toggle( state ) {
			this.open = state ? ( state === 'collapse' ? false : true ) : !this.open;
			this.$refs.parent.classList.toggle( 'lqd-is-active', this.open );
		},
		parent: {
			[ '@mouseenter' ]() {
				this.parentRect = this.$refs.parent.getBoundingClientRect();
				this.dropdownRect = this.$el.getBoundingClientRect();

				this.$el.classList.toggle('dropdown-anchor-bottom', this.parentRect.bottom + this.dropdownRect.height > window.innerHeight && this.parentRect.top - this.dropdownRect.height > 0);

				if ( this.triggerType === 'hover' ) {
					this.toggle( 'expand' );
				}
			},
			[ '@mouseleave' ](event) {
				if (
					this.triggerType !== 'hover' ||
					(
						event.relatedTarget === this.$refs.dropdown || this.$refs.dropdown.contains(event.relatedTarget)
					)
				) return;

				this.toggle( 'collapse' );
			},
			[ '@click.outside' ](event) {
				if ( event.target === this.$refs.dropdown || this.$refs.dropdown.contains(event.target) ) {
					return;
				}

				this.toggle( 'collapse' );
			},
		},
		trigger: {
			[ '@click.prevent' ]() {
				// we need to be able to toggle dropdown when focus/enter key is pressed
				// if (this.triggerType !== 'click') return;
				this.toggle();
			},
		},
		dropdown: {
			[ '@mouseleave' ]() {
				if ( triggerType !== 'hover' ) return;
				this.toggle( 'collapse' );
			},
			[':style']() {
				if ( !this.teleport ) return;

				const { top, bottom, left, right } = this.parentRect;
				const { width: dropWidth, height: dropHeight } = this.dropdownRect;
				const { innerWidth: vw, innerHeight: vh, scrollY } = window;
				const isRTL = document.dir === 'rtl' || document.documentElement.dir === 'rtl';

				// Vertical: default below parent, adjust if overflows
				let blockValue = bottom + scrollY;

				// If overflows bottom, try shifting up
				if (bottom + dropHeight > vh) {
					const shifted = vh - dropHeight + scrollY;
					blockValue = Math.max(scrollY, shifted);
				}

				// Horizontal positioning with preferredAnchor and overflow handling
				const anchorAtEnd = this.preferredAnchor === 'end';
				const inlineStart = isRTL ? vw - right : left;
				const inlineEnd = isRTL ? vw - left : right;

				const overflowsEnd = (anchorAtEnd ? inlineEnd : inlineStart) + dropWidth > vw;
				const overflowsStart = (anchorAtEnd ? inlineEnd : inlineStart) - dropWidth < 0;
				const shouldFlip = anchorAtEnd ? overflowsStart : overflowsEnd;

				const inlineAnchor = shouldFlip !== anchorAtEnd ? 'inset-inline-end' : 'inset-inline-start';
				const inlineValue = shouldFlip !== anchorAtEnd
					? Math.max(0, Math.min(vw - dropWidth, vw - inlineEnd))
					: Math.max(0, Math.min(vw - dropWidth, inlineStart));

				return {
					'inset-inline-start': inlineAnchor === 'inset-inline-start' ? `${inlineValue}px` : 'auto',
					'inset-inline-end': inlineAnchor === 'inset-inline-end' ? `${inlineValue}px` : 'auto',
					top: `calc(${blockValue}px + ${this.offsetY})`,
					bottom: 'auto',
				};
			}
		}
	} ) );

	// Notifications
	Alpine.store( 'notifications', {
		notifications: [],
		loading: false,
		add( notification ) {
			this.notifications.unshift( notification );
		},
		remove( index ) {
			this.notifications.splice( index, 1 );
		},
		markThenHref( notification ) {
			const index = this.notifications.indexOf( notification );
			if ( index === -1 ) return;
			var formData = new FormData();
			formData.append( 'id', notification.id );

			this.loading = true;

			$.ajax( {
				url: '/dashboard/notifications/mark-as-read',
				type: 'POST',
				data: formData,
				cache: false,
				contentType: false,
				processData: false,
				success: data => {
				},
				error: error => {
					console.error( error );
				},
				complete: () => {
					this.markAsRead( index );
					window.location = notification.link;
					this.loading = false;
				}
			} );
		},
		markAsRead( index ) {
			this.notifications = this.notifications.map( ( notification, i ) => {
				if ( i === index ) {
					notification.unread = false;
				}
				return notification;
			} );
		},
		markAllAsRead() {
			this.loading = true;
			$.ajax( {
				url: '/dashboard/notifications/mark-as-read',
				type: 'POST',
				success: response => {
					if ( response.success ) {
						this.notifications.forEach( ( notification, index ) => {
							this.markAsRead( index );
						} );
					}
				},
				error: error => {
					console.error( error );
				},
				complete: () => {
					this.loading = false;
				}
			} );
		},
		setNotifications( notifications ) {
			this.notifications = notifications;
		},
		hasUnread: function () {
			return this.notifications.some( notification => notification.unread );
		}
	} );
	Alpine.data( 'notifications', notifications => ( {
		notifications: notifications || [],
	} ) );

	// Focus Mode
	Alpine.store( 'focusMode', {
		active: Alpine.$persist( !!lqdFocusModeEnabled ).as( currentTheme + ':lqdFocusModeEnabled' ),
		toggle( state ) {
			this.active = state ? ( state === 'activate' ? true : false ) : !this.active;

			document.body.classList.toggle( 'focus-mode', this.active );
		},
	} );

	// Number Counter Component
	Alpine.data( 'numberCounter', ( { value = 0, options = {} } ) => ( {
		value: value,
		options: {
			delay: 0,
			...options
		},
		/**
		* @type {IntersectionObserver | null}
		*/
		io: null,
		numberWrappers: [],
		numberCols: [],
		numberAnimators: [],
		init() {
			this.$el.innerHTML = '';
			this.buildMarkup();
			this.setupIO();
		},
		updateValue( { value, options = {} } ) {
			if ( this.value === value ) return;

			this.value = value;
			this.options = {
				...this.options,
				...options
			};

			this.buildMarkup();
			this.setupIO();
		},
		buildMarkup() {
			const value = this.value.toString().split( '' );
			const currentNumberWrappers = this.$el.querySelectorAll( '.lqd-number-counter-numbers-wrap' );

			function buildNumberSpans() {
				return Array.from( { length: 10 }, ( _, i ) => `<span class="lqd-number-counter-number inline-flex h-full justify-center">${ i }</span>` ).join( '' );
			}

			const numberWrappers = value.map( ( value, index ) => {
				const isNumber = !isNaN( value );

				return `<span class="lqd-number-counter-numbers-wrap relative inline-flex h-full w-[1ch]" data-index="${ index }" data-value="${ value }"><span class="lqd-number-counter-numbers-col absolute start-0 top-[-0.25lh] inline-flex h-[1.5lh] w-full flex-col overflow-hidden py-[0.25lh]"><span class="lqd-number-counter-numbers-animator inline-flex w-full h-full flex-col" data-is-number="${ isNumber }" data-value="${ value }">${ isNumber ? buildNumberSpans() : value }</span></span></span>`;
			} );

			numberWrappers.forEach( ( wrapper, index ) => {
				const val = value[ index ];
				const existingEl = currentNumberWrappers[ index ];
				const isNumber = !isNaN( val );

				if ( existingEl ) {
					const animatorEl = existingEl.querySelector( '.lqd-number-counter-numbers-animator' );

					existingEl.setAttribute( 'data-value', val );

					animatorEl.setAttribute( 'data-value', val );
					animatorEl.setAttribute( 'data-is-number', isNumber );
					if ( animatorEl.getAttribute( 'data-is-number' ) === 'true' && isNumber ) {
						if ( animatorEl.innerHTML !== buildNumberSpans() ) {
							animatorEl.innerHTML = buildNumberSpans();
						}
					} else if ( animatorEl.innerHTML !== val ) {
						animatorEl.innerHTML = val;
					}

					return;
				}

				this.$el.insertAdjacentHTML( 'beforeend', wrapper );

				if ( currentNumberWrappers.length ) {
					const currentNumberWrapper = this.$el.querySelector( `.lqd-number-counter-numbers-wrap[data-index="${ index }"]` );

					currentNumberWrapper.animate( [
						{ translate: '0 0.25lh', opacity: 0 },
						{ translate: '0 0', opacity: 1 },
					], {
						duration: 250,
						easing: 'ease',
						fill: 'both'
					} );
				}
			} );

			// Remove extra currentNumberWrappers
			if ( currentNumberWrappers.length > value.length ) {
				for ( let i = value.length; i < currentNumberWrappers.length; i++ ) {
					currentNumberWrappers[ i ].animate( [
						{ translate: '0 -0.25lh', opacity: 0 },
					], {
						duration: 250,
						easing: 'ease',
						fill: 'both'
					} ).onfinish = () => {
						currentNumberWrappers[ i ].remove();
					};
				}
			}

			this.numberWrappers = this.$el.querySelectorAll( '.lqd-number-counter-numbers-wrap' );
			this.numberCols = this.$el.querySelectorAll( '.lqd-number-counter-numbers-col' );
			this.numberAnimators = this.$el.querySelectorAll( '.lqd-number-counter-numbers-animator' );
		},
		setupIO() {
			this.io = new IntersectionObserver( ( [ entry ], observer ) => {
				if ( entry.isIntersecting ) {
					observer.disconnect();
					this.animate();
				}
			} );

			this.io.observe( this.$el );
		},
		animate() {
			this.numberAnimators.forEach( el => {
				const isNumber = el.getAttribute( 'data-is-number' ) === 'true';

				if ( !isNumber ) return;

				const value = el.getAttribute( 'data-value' );

				el.animate( [
					// {
					// 	translate: '0 0',
					// },
					{
						translate: `0 ${ value * 100 * -1 }%`
					}
				], {
					duration: 800,
					delay: this.options.delay,
					easing: 'cubic-bezier(.47,1.09,.69,1.07)',
					fill: 'both'
				} );
			} );
		}
	} ) );

	// Shape Cutout
	Alpine.data( 'shapeCutout', () => ( {
		init() {
			this.onResize = this.onResize.bind( this );
			this.afterResize = debounce( this.afterResize.bind( this ), 1 );

			this.svgEl = this.$el.querySelector( 'svg' );

			if ( !this.svgEl ) return;

			this.svgObjects = this.svgEl.querySelectorAll( 'rect, circle, path, polygon' );

			this.events();
		},
		events() {
			$( window ).on( 'resize', this.onResize );

			this.resizeObserver = new ResizeObserver( () => {
				this.onResize();
			} );

			this.resizeObserver.observe( this.svgEl );
		},
		onResize() {
			this.changeObjAttr( '-' );

			this.afterResize();
		},
		afterResize() {
			this.changeObjAttr( '+' );
		},
		changeObjAttr( operator ) {
			this.svgObjects.forEach( obj => {
				if ( obj.hasAttribute( 'x' ) ) {
					obj.setAttribute( 'x', parseFloat( parseFloat( obj.getAttribute( 'x' ) ) + operator + '1' ) );
				} else if ( obj.hasAttribute( 'width' ) ) {
					obj.setAttribute( 'width', parseFloat( parseFloat( obj.getAttribute( 'width' ) ) + operator + '1' ) );
				} else if ( obj.hasAttribute( 'cx' ) ) {
					obj.setAttribute( 'cx', parseFloat( parseFloat( obj.getAttribute( 'cx' ) ) + operator + '1' ) );
				} else if ( obj.hasAttribute( 'r' ) ) {
					obj.setAttribute( 'r', parseFloat( parseFloat( obj.getAttribute( 'r' ) ) + operator + '1' ) );
				}
			} );
		}
	} ) );

	// Marquee
	Alpine.data( 'marquee', ( options = {} ) => ( {
		maxWidth: 0,
		position: 0,
		options: {
			direction: -1,
			speed: 0.5,
			pauseOnHover: false,
			...options
		},
		async init() {
			this.direction = this.options.direction;
			this.cellWidths = [];
			this.cellHeights = [];
			this.viewportEl = this.$el.querySelector( '.lqd-marquee-viewport' );
			this.sliderEl = this.$el.querySelector( '.lqd-marquee-slider' );
			this.cells = this.sliderEl.querySelectorAll( '.lqd-marquee-cell' );
			this.sliderElStyles = window.getComputedStyle( this.sliderEl );
			this.maxWidth = 0;
			this.maxHeight = 0;

			this.onResize = debounce( this.onResize.bind( this ), 450 );

			await document.fonts.ready;

			this.sizing();

			this.startAnimation();
		},
		sizing() {
			for ( let i = 0; i < this.cells.length; i++ ) {
				this.cellHeights.push( this.cells[ i ].offsetHeight );
				this.cellWidths.push( this.cells[ i ].offsetWidth );
			}

			this.maxHeight = Math.max( ...this.cellHeights );
			this.maxWidth = this.cellWidths.reduce( ( acc, width ) => acc + width, 0 );

			this.maxWidth += parseInt( this.sliderElStyles.paddingLeft ) + parseInt( this.sliderElStyles.paddingRight );
			// this.maxWidth += parseInt(this.sliderElStyles.marginLeft) + parseInt(this.sliderElStyles.marginRight);
			// this.maxWidth += parseInt(this.sliderElStyles.borderLeftWidth) + parseInt(this.sliderElStyles.borderRightWidth);
			this.maxWidth += parseInt( this.sliderElStyles.gap ) * ( this.cells.length - 1 );

			this.viewportEl.style.height = `${ this.maxHeight + parseInt( this.sliderElStyles.paddingTop ) + parseInt( this.sliderElStyles.paddingBottom ) }px`;
			this.sliderEl.classList.add( 'absolute', 'top-0', 'left-0', 'w-full', 'h-full' );

			this.maxWidth -= this.viewportEl.offsetWidth;
		},
		startAnimation() {
			this.isAnimating = true;

			if ( this.options.pauseOnHover ) {
				this.sliderEl.addEventListener( 'pointerenter', () => {
					this.isAnimating = false;
				} );

				this.sliderEl.addEventListener( 'pointerleave', () => {
					this.isAnimating = true;
				} );
			}

			const animate = () => {
				if ( this.isAnimating ) {
					this.position += this.options.speed * this.direction;

					if ( this.position <= -this.maxWidth ) {
						this.direction = 1;
					} else if ( this.position >= 0 ) {
						this.direction = -1;
					}

					this.sliderEl.style.transform = `translateX(${ this.position }px)`;
				}

				requestAnimationFrame( animate );
			};

			requestAnimationFrame( animate );
		},
		onResize() {
			this.sizing();
		}
	} ) );

	/**
	 * Maruqee V2
	 * @requires GSAP
	 * @requires Observer
	 * @requires ScrollTrigger
	 */
	Alpine.data( 'marqueev2', (options = {}) => ( {
		cells: [],
		timeline: null,
		IO: null,
		active: false,
		options: {
			cellsSelector: '.lqd-marquee-cell',
			...options
		},
		init() {
			this.cells = this.$el.querySelectorAll(this.options.cellsSelector);

			this.timeline = this.horizontalLoop(this.cells, {
				repeat: -1,
				paddingRight: 24,
			});

			this.IO = new IntersectionObserver(([ entry ]) => {
				this.active = entry.isIntersecting;

				this.cells.forEach(cell => {
					cell.style.willChange = this.active ? 'transform' : 'auto';
				});
			}).observe(this.$el);

			this.createObserver();
		},

		createObserver() {
			Observer.create({
				onChangeY: observer => {
					// let factor = this.active ? 2.5 : 0;
					let factor = 1.5;
					if (observer.deltaY < 0) {
						factor *= -1;
					}

					gsap.timeline({
						defaults: {
							ease: 'none',
						}
					})
						.to(this.timeline, { timeScale: factor * 2.5, duration: 0.2, overwrite: true, })
						.to(this.timeline, { timeScale: factor / 2.5, duration: 1 }, '+=0.3');
				}
			});
		},

		/*
			https://gsap.com/docs/v3/HelperFunctions/helpers/seamlessLoop/
			This helper function makes a group of elements animate along the x-axis in a seamless, responsive loop.

			Features:
			- Uses xPercent so that even if the widths change (like if the window gets resized), it should still work in most cases.
			- When each item animates to the left or right enough, it will loop back to the other side
			- Optionally pass in a config object with values like "speed" (default: 1, which travels at roughly 100 pixels per second), paused (boolean),  repeat, reversed, and paddingRight.
			- The returned timeline will have the following methods added to it:
			- next() - animates to the next element using a timeline.tweenTo() which it returns. You can pass in a vars object to control duration, easing, etc.
			- previous() - animates to the previous element using a timeline.tweenTo() which it returns. You can pass in a vars object to control duration, easing, etc.
			- toIndex() - pass in a zero-based index value of the element that it should animate to, and optionally pass in a vars object to control duration, easing, etc. Always goes in the shortest direction
			- current() - returns the current index (if an animation is in-progress, it reflects the final index)
			- times - an Array of the times on the timeline where each element hits the "starting" spot. There's also a label added accordingly, so "label1" is when the 2nd element reaches the start.
		*/
		horizontalLoop(items, config) {
			items = gsap.utils.toArray(items);
			config = config || {};
			let tl = gsap.timeline({ repeat: config.repeat, paused: config.paused, defaults: { ease: 'none' }, onReverseComplete: () => tl.totalTime(tl.rawTime() + tl.duration() * 100) }),
				length = items.length,
				startX = items[0].offsetLeft,
				times = [],
				widths = [],
				xPercents = [],
				curIndex = 0,
				pixelsPerSecond = (config.speed || 1) * 100,
				snap = config.snap === false ? v => v : gsap.utils.snap(config.snap || 1), // some browsers shift by a pixel to accommodate flex layouts, so for example if width is 20% the first element's width might be 242px, and the next 243px, alternating back and forth. So we snap to 5 percentage points to make things look more natural
				totalWidth, curX, distanceToStart, distanceToLoop, item, i;
			gsap.set(items, { // convert "x" to "xPercent" to make things responsive, and populate the widths/xPercents Arrays to make lookups faster.
				xPercent: (i, el) => {
					let w = widths[i] = parseFloat(gsap.getProperty(el, 'width', 'px'));
					xPercents[i] = snap(parseFloat(gsap.getProperty(el, 'x', 'px')) / w * 100 + gsap.getProperty(el, 'xPercent'));
					return xPercents[i];
				}
			});
			gsap.set(items, { x: 0 });
			totalWidth = items[length-1].offsetLeft + xPercents[length-1] / 100 * widths[length-1] - startX + items[length-1].offsetWidth * gsap.getProperty(items[length-1], 'scaleX') + (parseFloat(config.paddingRight) || 0);
			for (i = 0; i < length; i++) {
				item = items[i];
				curX = xPercents[i] / 100 * widths[i];
				distanceToStart = item.offsetLeft + curX - startX;
				distanceToLoop = distanceToStart + widths[i] * gsap.getProperty(item, 'scaleX');
				tl.to(item, { xPercent: snap((curX - distanceToLoop) / widths[i] * 100), duration: distanceToLoop / pixelsPerSecond }, 0)
					.fromTo(item, { xPercent: snap((curX - distanceToLoop + totalWidth) / widths[i] * 100) }, { xPercent: xPercents[i], duration: (curX - distanceToLoop + totalWidth - curX) / pixelsPerSecond, immediateRender: false }, distanceToLoop / pixelsPerSecond)
					.add('label' + i, distanceToStart / pixelsPerSecond);
				times[i] = distanceToStart / pixelsPerSecond;
			}
			function toIndex(index, vars) {
				vars = vars || {};
				(Math.abs(index - curIndex) > length / 2) && (index += index > curIndex ? -length : length); // always go in the shortest direction
				let newIndex = gsap.utils.wrap(0, length, index),
					time = times[newIndex];
				if (time > tl.time() !== index > curIndex) { // if we're wrapping the timeline's playhead, make the proper adjustments
					vars.modifiers = { time: gsap.utils.wrap(0, tl.duration()) };
					time += tl.duration() * (index > curIndex ? 1 : -1);
				}
				curIndex = newIndex;
				vars.overwrite = true;
				return tl.tweenTo(time, vars);
			}
			tl.next = vars => toIndex(curIndex+1, vars);
			tl.previous = vars => toIndex(curIndex-1, vars);
			tl.current = () => curIndex;
			tl.toIndex = (index, vars) => toIndex(index, vars);
			tl.times = times;
			tl.progress(1, true).progress(0, true); // pre-render for performance
			if (config.reversed) {
				tl.vars.onReverseComplete();
				tl.reverse();
			}
			return tl;
		}
	} ) );

	// Curtain
	Alpine.data( 'curtain', ( id = 'curtain', options = {} ) => ( {
		id: id,
		activeCurtain: 0,
		options: {
			itemsSelector: '.lqd-curtain-item',
			contentSelector: '.lqd-curtain-item-content',
			contentWidthOuter: '.lqd-curtain-item-content-width-outer',
			contentWidthInner: '.lqd-curtain-item-content-width-inner',
			activeClassname: 'lqd-curtain-item-active',
			inactiveClassname: 'lqd-curtain-item-inactive',
			duration: 0.65,
			ease: 'cubic-bezier(0.23, 1, 0.320, 1)',
			trigger: 'pointerenter',
			...options
		},
		init() {
			this.items = [ ...this.$el.querySelectorAll( this.options.itemsSelector ) ];

			if ( !this.items.length ) return;

			this.onElementActive = this.onElementActive.bind( this );
			this.onWindowResize = debounce( this.onWindowResize.bind( this ), 450 );

			this.setActiveCurtain();
			this.setActiveElement();
			this.setActiveContentWidth();
			this.events();
		},
		events() {
			const { trigger } = this.options;
			const onElementActive = throttle( this.onElementActive, 50, { leading: true, trailing: false } );

			this.items.forEach( item => {
				item.addEventListener( trigger, onElementActive );
			} );

			window.addEventListener( 'resize', this.onWindowResize );
		},
		setActiveCurtain() {
			this.activeCurtain = this.items.findIndex( item => item.classList.contains( this.options.activeClassname ) );

			this.$dispatch( `curtain-changed-${ this.id }`, { activeCurtain: this.activeCurtain } );
		},
		setActiveElement() {
			this.activeElement = this.items[ this.activeCurtain ];
		},
		setActiveContentWidth() {
			if ( !this.getElDirection().includes( 'row' ) ) return;

			const contentWidthOuter = this.activeElement.querySelector( this.options.contentWidthOuter );
			const activeElContentWidth = contentWidthOuter.offsetWidth;

			this.$el.style.setProperty( '--active-width', `${ activeElContentWidth }px` );
		},
		onElementActive( event ) {
			const { activeClassname, inactiveClassname } = this.options;
			const activeElement = event.currentTarget;

			this.items.forEach( item => {
				item.classList.remove( activeClassname );
				item.classList.add( inactiveClassname );
			} );

			activeElement.classList.remove( inactiveClassname );
			activeElement.classList.add( activeClassname );

			this.setActiveCurtain();
			this.setActiveElement();
		},
		/**
		*
		* @returns {string} - The flex-direction of the element
		*/
		getElDirection() {
			const elStyles = window.getComputedStyle( this.activeElement );
			return elStyles.flexDirection;
		},
		onWindowResize() {
			this.setActiveContentWidth();
		}
	} ) );

	// Slideshow
	Alpine.data( 'slideshow', ( id = 'slideshow', totalSlides = 0, options = {} ) => ( {
		activeSlide: 0,
		totalSlides: totalSlides,
		id: id,
		options: {
			...options
		},
		init() {
			this.setActiveSlide = this.setActiveSlide.bind( this );
		},
		/**
		 * @param {number | '>' | '<'} index
		 */
		setActiveSlide( index ) {
			if ( index === '>' ) {
				index = this.activeSlide + 1;
			} else if ( index === '<' ) {
				index = this.activeSlide - 1;
			}

			if ( index < 0 ) {
				index = this.totalSlides - 1;
			} else if ( index >= this.totalSlides ) {
				index = 0;
			}

			this.activeSlide = index;

			this.$dispatch( `slide-changed-${ this.id }`, { activeSlide: this.activeSlide } );
		}
	} ) );

	// Dynamic Input
	Alpine.data( 'dynamicInput', ( options = { relativeValue: false, value: 0, min: null, max: null, step: 1, onInput: null } ) => ( {
		value: options.value ?? 0,
		_relativeValue: options.relativeValue,
		originalRelativeValue: null,
		min: options.min,
		max: options.max,
		step: options.step ?? 1,
		onInputFn: options.onInput,
		prevMouseX: null,
		overlay: null,
		mouseDown: false,
		changingDelta: 0,
		prevVal: null,

		get relativeValue() {
			const opt = this._relativeValue;
			return typeof opt === 'function' ? opt() : opt;
		},

		set relativeValue( value ) {
			this._relativeValue = value;
		},

		init() {
			this.onMouseDown = this.onMouseDown.bind( this );
			this.onMouseMove = this.onMouseMove.bind( this );
			this.onMouseUp = this.onMouseUp.bind( this );
			this.onKeyDown = this.onKeyDown.bind( this );
			this.onInput = this.onInput.bind( this );

			this.revertBackRelativeValue = _.throttle( this.revertBackRelativeValue.bind( this ), 150, { leading: false } );

			if ( this.value != null ) {
				this.updateValue( this.value );
			}

			this.events();

			this.$watch( 'mouseDown', isMouseDown => {
				this.$el.classList.toggle( 'dragging', isMouseDown );
			} );
		},
		events() {
			const dynamicLabel = this.$refs.dynamicLabel;
			const dynamicInput = this.$refs.dynamicInput;

			if ( !dynamicLabel || !dynamicInput ) return;

			dynamicLabel.addEventListener( 'mousedown', this.onMouseDown );
			window.addEventListener( 'mousemove', this.onMouseMove );
			window.addEventListener( 'mouseup', this.onMouseUp );

			dynamicInput.addEventListener( 'keydown', this.onKeyDown );
			dynamicInput.addEventListener( 'input', this.onInput );
		},
		updateInputValue( value, dispatchInput = true ) {
			const dynamicInput = this.$refs.dynamicInput;

			if ( !dynamicInput ) return;

			if ( !isNaN( value ) ) {
				// Get decimal precision from step and value
				let decimalPrecision = 0;
				const stepStr = this.step.toString();
				const valueStr = value.toString();

				// Check step precision
				if ( stepStr.includes( '.' ) ) {
					decimalPrecision = Math.min( 2, stepStr.split( '.' )[ 1 ].length );
				}

				// Check value precision
				if ( valueStr.includes( '.' ) ) {
					decimalPrecision = Math.min( 2, valueStr.split( '.' )[ 1 ].length );
				}

				// Format value with proper decimal places, max 2
				if ( decimalPrecision > 0 ) {
					value = parseFloat( value ).toFixed( decimalPrecision );
				}
			}

			dynamicInput.value = value;

			dispatchInput && dynamicInput.dispatchEvent( new Event( 'input', { bubbles: true } ) );
		},
		updateValue( value, updateInput = true, dispatchInput = true ) {
			if ( value == null ) return;

			const dynamicInput = this.$refs.dynamicInput;

			if ( !dynamicInput ) return;

			if ( this.relativeValue && isNaN( value ) ) {
				this.value = value;
				this.updateInputValue( this.value, false );

				return;
			}

			let val = parseFloat( value );

			if ( !this.relativeValue && this.min != null && val < this.min ) {
				val = this.min;
			}
			if ( !this.relativeValue && this.max != null && val > this.max ) {
				val = this.max;
			}

			this.value = val;

			updateInput && this.updateInputValue( this.value, dispatchInput );
		},
		onMouseDown( event ) {
			const dynamicInput = this.$refs.dynamicInput;

			this.mouseDown = true;
			this.prevVal = dynamicInput.value;

			// Prevent text selection during dragging
			event.preventDefault();

			if ( !this.overlay ) {
				this.overlay = document.createElement( 'div' );
				this.overlay.classList.add( 'fixed', 'top-0', 'start-0', 'w-screen', 'h-screen', 'z-10' );
				this.overlay.style.cursor = 'ew-resize';
				document.body.appendChild( this.overlay );
			}
		},
		onMouseMove( event ) {
			if ( !this.mouseDown ) return;

			if ( !this.prevMouseX && this.prevMouseX !== 0 ) {
				this.prevMouseX = event.clientX;
				return;
			}

			const mouseX = event.clientX;
			const deltaX = mouseX - this.prevMouseX;
			const shiftPressed = event.shiftKey;
			const metaKey = event.metaKey || event.ctrlKey;
			const sensitivity = this.step * ( shiftPressed ? 10 : metaKey ? 0.1 : 1 );

			if ( deltaX !== 0 ) {
				const changeAmount = ( deltaX > 0 ? 1 : -1 ) * sensitivity;
				const valueIsNumber = !isNaN( parseFloat( this.value ) );
				let val = ( valueIsNumber ? parseFloat( this.value ) : 0 ) + changeAmount;

				if ( this.relativeValue ) {
					val = changeAmount;
				}

				this.changingDelta += changeAmount;

				this.updateValue( val );

				this.prevMouseX = mouseX;
			}
		},
		onMouseUp() {
			if ( !this.mouseDown ) return;

			if ( this.relativeValue ) {
				this.updateValue(
					isNaN( this.prevVal ) ? this.prevVal : parseFloat( this.prevVal || 0 ) + this.changingDelta,
					true,
					false
				);
			}

			this.mouseDown = false;
			this.prevMouseX = null;
			this.changingDelta = 0;
			this.prevVal = null;

			if ( this.overlay ) {
				document.body.removeChild( this.overlay );
				this.overlay = null;
			}
		},
		onKeyDown( event ) {
			if ( event.key === 'Enter' || event.key === 'Tab' ) {
				this.updateValue( this.calculateExpression() );

				if ( event.key === 'Tab' ) {
					// Allow default tab behavior to continue (moving to next input)
					return true;
				} else {
					// Prevent form submission on Enter
					event.preventDefault();
				}
			} else if ( event.key === 'ArrowUp' || event.key === 'ArrowDown' ) {
				event.preventDefault();

				const shiftPressed = event.shiftKey;
				const metaKey = event.metaKey || event.ctrlKey;
				const step = this.step * ( shiftPressed ? 10 : metaKey ? 0.1 : 1 );
				const changeAmount = event.key === 'ArrowUp' ? step : -step;
				const inputValue = event.target.value;
				const valueIsNumber = !isNaN( parseFloat( inputValue ) );
				let val = ( valueIsNumber ? parseFloat( inputValue ) : 0 ) + changeAmount;

				if ( this.relativeValue ) {
					val = changeAmount;
				}

				this.updateValue( val );
			}
		},
		onInput( event ) {
			this.updateValue( event.target.value, false );

			if ( typeof this.onInputFn === 'function' ) {
				this.onInputFn.call( this, this.value );
			}
		},
		calculateExpression() {
			const inputValue = this.$refs.dynamicInput.value.trim();
			let value = inputValue;

			if ( !inputValue ) return;

			// Check if the input contains an expression and ends with a number or closing parenthesis
			if ( /[-+*/().]/.test( inputValue ) && /[\d)]$/.test( inputValue ) ) {
				try {
					// Make sure the expression is complete before evaluating
					if ( this.isValidExpression( inputValue ) ) {
						// Use Function constructor to safely evaluate the expression
						const result = Function( '"use strict"; return (' + inputValue + ')' )();

						// Check if result is a valid number
						if ( !isNaN( result ) && isFinite( result ) ) {
							value = result;
							return;
						}
					}
				} catch ( error ) {
					// If expression evaluation fails, keep the current input value
					console.log( 'Invalid expression:', error );
				}
			}

			return value;
		},
		isValidExpression( expr ) {
			// Check for balanced parentheses
			let parenCount = 0;
			for ( let i = 0; i < expr.length; i++ ) {
				if ( expr[ i ] === '(' ) parenCount++;
				if ( expr[ i ] === ')' ) parenCount--;
				if ( parenCount < 0 ) return false;
			}
			if ( parenCount !== 0 ) return false;

			// Check for invalid sequences of operators
			if ( /[+\-*/]{2,}/.test( expr ) ) return false;

			// Check if expression starts with an operator (except minus)
			if ( /^[+*/]/.test( expr ) ) return false;

			// Check if expression ends with an operator
			if ( /[+\-*/]$/.test( expr ) ) return false;

			return true;
		},
		revertBackRelativeValue() {
			if ( !this.originalRelativeValue ) return;

			this.relativeValue = this.originalRelativeValue;

			this.originalRelativeValue = null;
		}
	} ) );

	/**
	 * Split Text
	 * @requires GSAP
	 * @requires SplitText
	 */
	Alpine.data( 'splitText', ( options = {} ) => ( {
		splitText: null,
		options: {
			type: 'words',
			tag: 'span',
			charsClass: 'lqd-split-unit lqd-split-char',
			wordsClass: 'lqd-split-unit lqd-split-word',
			linesClass: 'lqd-split-unit lqd-split-line',
			...options,
		},
		init() {
			this.splitText = new SplitText( this.$el, this.options );

			const wordsLength = this.splitText.words.length;

			this.splitText.words.forEach( ( word, i ) => {
				word.setAttribute( 'data-index', i );
				word.setAttribute( 'data-last-index', wordsLength - 1 - i );

				word.style.setProperty( '--word-index', i );
				word.style.setProperty( '--word-last-index', wordsLength - 1 - i );
			} );

			this.$dispatch( 'split-text-done', { splitText: this.splitText } );
		},
	} ) );

	Alpine.data( 'liquidColorPicker', ( options = { colorVal: null, onPick: null } ) => ( {
		_colorVal: options.colorVal,
		picker: null,
		onPick: options.onPick,

		get colorVal() {
			return this._colorVal;
		},

		set colorVal( color ) {
			this._colorVal = color;
		},

		init() {
			this.checkDarkMode = this.checkDarkMode.bind( this );

			this.checkDarkMode();
			this.initColorPicker();
			this.events();
		},

		initColorPicker() {
			this.$refs.colorInput.setAttribute( 'type', 'text' );
			this.picker = new ColorPicker( this.$refs.colorInputWrap ?? this.$el, {
				color: this.colorVal,
				submitMode: 'instant',
				showClearButton: true
			} );
		},

		events() {
			this.$watch( '$store.darkMode.on', () => {
				this.checkDarkMode();
			} );

			this.picker.on( 'pick', color => {
				this.colorVal = color;

				if ( typeof this.onPick === 'function' ) {
					this.onPick.call( this, color );
				}
				if ( this.$refs.colorInput ) {
					this.$refs.colorInput.value = color;
					this.$refs.colorInput.dispatchEvent( new Event( 'input', { bubbles: true } ) );
				}
			} );
		},

		checkDarkMode() {
			const darkMode = localStorage.getItem( 'lqdDarkMode' ) == 'true';

			document.documentElement.setAttribute( 'data-cp-theme', darkMode ? 'dark' : 'light' );
			document.documentElement.setAttribute( 'data-bs-theme', darkMode ? 'dark' : 'light' );
		}
	} ) );

	/**
	 * @requires GSAP
	 * @requires ScrollTrigger
	 * @requires SplitText
	 */
	Alpine.data('liquidTextReveal', ({ splitEl = null, splitType = 'chars', start = 'top bottom', end = 'center 65%', animateFrom = { opacity: 0.2 }, animateTo = { opacity: 1 } }) => ({
		splitType: splitType === 'chars' ? 'chars,words' : splitType,
		start: start,
		end: end,
		animateFrom: animateFrom,
		animateTo: animateTo,
		splittedText: null,
		splitEl: null,

		getSplitEl() {
			return splitEl ?? this.$el;
		},

		init() {
			this.onTextSplitted = this.onTextSplitted.bind(this);

			this.initSplitText();
		},
		initSplitText() {
			SplitText.create(this.getSplitEl(), {
				autoSplit: true,
				onSplit: this.onTextSplitted
			});
		},
		getAnimations() {
			const els = this.splittedText[this.splitType === 'chars,words' ? 'chars' : this.splitType];

			return gsap.fromTo(els,
				{ ...this.animateFrom },
				{
					stagger: 0.1,
					...this.animateTo
				}
			);
		},
		onTextSplitted(splittedText) {
			this.splittedText = splittedText;

			ScrollTrigger.create({
				animation: this.getAnimations(),
				trigger: this.getSplitEl(),
				scrub: true,
				start: this.start,
				end: this.end,
			});
		}
	}));

	/**
	 * @requires ScrollSmoother
	 */
	Alpine.data('liquidScrollSmooth', () => ({
		init() {
			ScrollSmoother.create({
				smooth: 1,
				effects: true,
				smoothTouch: 0.1
			});
		}
	}));

	Alpine.data('updateAvailable', ({ routes = {} }) => ({
		routes: {
			...routes || {}
		},
		route: '#',
		isAvailable: false,
		isVersionUpdateAvailable: false,
		isExtensionUpdateAvailable: false,
		updateAvailableExtensions: [],

		init() {
			this.checkAvailability();
		},
		async checkAvailability() {
			const res = await fetch(this.routes.check);

			if (!res.ok) {
				console.error('Network error: check update availablity');
				return;
			}

			const resData = await res.json();

			this.isVersionUpdateAvailable = resData.versionUpdateAvailable;
			this.isExtensionUpdateAvailable = resData.extensionUpdateAvailable;
			this.updateAvailableExtensions = resData.updateAvailableExtensions;
			this.isAvailable = this.isVersionUpdateAvailable || this.isExtensionUpdateAvailable;

			if (this.isVersionUpdateAvailable) {
				this.route = this.routes.appUpdate;
			} else if (this.isExtensionUpdateAvailable) {
				this.route = this.routes.extensionUpdate;
			}
		}
	}));

	Alpine.data('liquidMegamenu', () => ({
		liEl: null,
		header: null,
		posAppliedClassname: 'sub-pos-applied',
		prevMouseX: null,
		init() {
			this.onWindowMouseMove = this.onWindowMouseMove.bind(this);

			this.liEl = this.$el.closest('li');
			this.header = this.$el.closest('.site-header');
			this.nav = this.$el.closest('.site-header-nav');

			this.position();
		},
		position() {
			this.liEl.classList.remove(this.posAppliedClassname);

			if (window.innerWidth < 992) {
				return this.liEl.classList.add(this.posAppliedClassname);
			}

			const liRect = this.liEl.getBoundingClientRect();
			const elRect = this.$el.getBoundingClientRect();
			const navRect = this.nav.getBoundingClientRect();
			const viewportWidth = window.innerWidth;
			const liBottomToHeaderBottom = navRect.bottom - liRect.bottom;

			// Center the element in the viewport
			let diff = (viewportWidth - elRect.width) / 2;

			// Adjust if the element's left edge goes beyond the viewport's left edge
			if (diff < 0) {
				diff = 0;
			}

			// Adjust if the element goes beyond the right boundary of the viewport
			if (diff + elRect.width > viewportWidth) {
				diff = viewportWidth - elRect.width;
			}

			// Adjust if the element goes beyond the right boundary of li element
			if (diff > liRect.right) {
				diff = liRect.left - 20;
			}

			this.$el.style.left = `${diff - liRect.left}px`;

			this.liEl.style.setProperty('--sub-offset', `${liBottomToHeaderBottom}px`);

			this.liEl.classList.add(this.posAppliedClassname);
		},
		onWindowMouseMove(event) {
			if (window.innerWidth < 992) {
				return;
			}
			if (this.prevMouseX !== undefined) {
				if (event.clientX > this.prevMouseX) {
					this.$el.setAttribute('data-direction', 'right');
				} else if (event.clientX < this.prevMouseX) {
					this.$el.setAttribute('data-direction', 'left');
				}
			}
			this.prevMouseX = event.clientX;
		}
	}));

	// OpenAI Realtime
	Alpine.data( 'openaiRealtime', openaiRealtime );

	// Elevenlabs Realtime
	Alpine.data( 'elevenlabsRealtime', elevenlabsRealtime );

	// tiptap Editor
	Alpine.data( 'tiptapEditor', tiptapEditor );

	// Advanced Image Editor
	Alpine.data( 'advancedImageEditor', advancedImageEditor );

	// Creative Suite
	Alpine.data( 'creativeSuite', creativeSuite );

	Alpine.data( 'advancedImageEditor', advancedImageEditor );

	// Customizer
	Alpine.data( 'lqdCustomizer', lqdCustomizer );
	Alpine.data( 'lqdCustomizerFontPicker', lqdCustomizerFontPicker );
} );

Livewire.start();
