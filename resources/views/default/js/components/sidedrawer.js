/**
 * @type {import('alpinejs').AlpineComponent}
 * @param {Object} options
 */
export function lqdSidedrawer() {
	return ({
		_open: false,
		options: {},

		get sidedrawerOpen() {
			return this._open;
		},

		set sidedrawerOpen(state) {
			const st = !!state;

			if ( st === this._open ) return;

			this._open = st;
		},
	});
}
