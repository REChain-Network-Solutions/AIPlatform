export default function liquidSelectBox( {
	value = null,
	optionsScriptId = null,
	placeholder = '',
	pageSize = 25,
} = {} ) {
	let initialOptions = [];

	if ( optionsScriptId ) {
		const el = document.getElementById( optionsScriptId );
		if ( el ) {
			try {
				initialOptions = JSON.parse( el.textContent );
			} catch ( e ) {
				console.error( 'liquidSelectBox: failed to parse options JSON', e );
			}
		}
	}

	const labels = {};
	initialOptions.forEach( ( option ) => {
		labels[ option.value ] = option.label;
	} );

	return {
		selected: value,
		query: '',
		options: initialOptions,
		labels,
		placeholder,
		pageSize,
		visibleCount: pageSize,
		_loadMoreObserver: null,

		init() {
			this.$watch( 'query', () => {
				this.visibleCount = this.pageSize;
			} );

			this.$nextTick( () => this.observeLoadMore() );
		},

		destroy() {
			if ( this._loadMoreObserver ) {
				this._loadMoreObserver.disconnect();
				this._loadMoreObserver = null;
			}
		},

		observeLoadMore() {
			const sentinel = this.$refs.loadMoreSentinel;
			if ( !sentinel ) return;

			const root = sentinel.closest( '.lqd-dropdown-dropdown-content' ) || null;

			this._loadMoreObserver = new IntersectionObserver( ( entries ) => {
				if ( !entries[ 0 ].isIntersecting ) return;
				if ( this.visibleCount >= this.filteredOptions.length ) return;
				this.visibleCount = Math.min(
					this.visibleCount + this.pageSize,
					this.filteredOptions.length,
				);
			}, { root, rootMargin: '100px' } );

			this._loadMoreObserver.observe( sentinel );
		},

		registerOption( value, label ) {
			this.labels[ value ] = label;
		},

		select( value ) {
			this.selected = value;
		},

		matches( value, label ) {
			if ( !this.query ) return true;
			return label.toLowerCase().includes( this.query.toLowerCase() );
		},

		get filteredOptions() {
			if ( !this.query ) return this.options;
			const q = this.query.toLowerCase();
			return this.options.filter( ( o ) => o.label.toLowerCase().includes( q ) );
		},

		get visibleOptions() {
			return this.filteredOptions.slice( 0, this.visibleCount );
		},

		get activeLabel() {
			if (
				this.selected !== null
				&& this.selected !== undefined
				&& this.labels[ this.selected ] !== undefined
			) {
				return this.labels[ this.selected ];
			}
			return this.placeholder;
		},

		get hasMatches() {
			if ( !this.query ) return true;
			const q = this.query.toLowerCase();
			return Object.values( this.labels ).some( ( l ) => l.toLowerCase().includes( q ) );
		},
	};
}
