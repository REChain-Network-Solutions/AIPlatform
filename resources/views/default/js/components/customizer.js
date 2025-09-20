import { generateColorRamp, generateColorRampWithCurve, colorUtils } from 'rampensau';

export function lqdCustomizer() {
	return ({
		options: {},
		styleTag: document.querySelector('#lqd-customizer-style'),
		darkMode: document.body.classList.contains('theme-dark'),
		rootStyles: null,
		styleString: '',
		lqdFontPreview: [],
		googleFontsList: [],
		showingGoogleFonts: [],
		fontsPaginationCurrent: 20,
		fontsPaginationLimit: 20,
		loadingFontPreviewQueue: [],
		localStorageKey: 'lqdCustomizerStyle',
		cssSelectors: {
			light: ':root',
			dark: '.theme-dark',
		},

		init() {

			const currentTheme = document.body.getAttribute('data-theme');

			this.localStorageKey = currentTheme + ':' + this.localStorageKey;
			this.onImportStyles = _.throttle(this.onImportStyles.bind(this), 300, {
				leading: false
			});
			this.loadGoogleFontPreview = _.throttle(this.loadGoogleFontPreview.bind(this), 300, {
				leading: false
			});

			this.options = window.lqdCustomizerOptions;
			this.googleFontsList = window.lqdGoogleFontsList;
			this.currentEdits = this.readFromLocalStorage();

			const isDark = document.body.classList.contains('theme-dark');

			this.rootStyles = getComputedStyle(isDark ? document.body : document.documentElement);

			Object.entries(this.options).forEach(([ key ]) => {
				if (this.options[key].type === 'color') {
					this.options[key].values = {};
					this.options[key].values.light = {};
					this.options[key].values.dark = {};
				}
			});

			this.$watch('$store.darkMode.on', isDark => {
				this.darkMode = isDark;
				this.rootStyles = getComputedStyle(isDark ? document.body : document.documentElement);
			});

			this.$watch('options', this.updateOutputs.bind(this));

			this.showingGoogleFonts = this.googleFontsList.slice(0, this.fontsPaginationLimit);

			if ( this.currentEdits )  {
				this.onImportStyles(this.currentEdits);
			}
		},

		setStyleStrings() {
			let lightStyleString = '';
			let darkStyleString = '';

			Object.entries(this.options).forEach(([ key, option ]) => {
				const isFont = option.type === 'font';
				const cssVar = option.cssVar;
				let lightValue = option.value;
				let darkValue = null;

				if ( option.values?.light ) {
					lightValue = option.values.light.twHsl;
				}

				if ( option.values?.dark ) {
					darkValue = option.values.dark.twHsl;
				}

				if (lightValue) {
					lightStyleString += `\t${cssVar}: ${isFont ? `'${lightValue}'` : lightValue };\n`;
				}

				if (darkValue) {
					darkStyleString += `\t${cssVar}: ${darkValue};\n`;
				}
			});

			if ( !lightStyleString.length && !darkStyleString.length ) {
				return this.styleString = '';
			}

			this.styleString = `${this.cssSelectors.light} {\n${lightStyleString}}`;

			if ( darkStyleString.length ) {
				this.styleString += `\n${this.cssSelectors.dark} {\n${darkStyleString}}`;
			}
		},

		updateOutputs() {
			this.setStyleStrings();
			this.fillStyleTag();
			this.writeToLocalStorage();
		},

		fillStyleTag() {
			if ( !this.styleTag ) {
				return console.error('Style tag not found');
			}

			this.styleTag.innerText = this.styleString.replace(/[\n\t]/g, '');
		},

		onColorInput({ key, color, manipulations = [], assosiatedKeys = [] }) {
			if (this.options[key]?.type !== 'color') return;

			const parsedColor = this.parseColor({ color, format: [ 'twHsl', 'hex' ], manipulations });
			const darkOrLight = this.darkMode ? 'dark' : 'light';

			this.options[key].values[darkOrLight] = {
				twHsl: parsedColor.twHsl,
				hex: parsedColor.hex,
			};

			if ( assosiatedKeys.length ) {
				assosiatedKeys.forEach(({ key: assosiatedColorKey, color: assosiatedColorValue, manipulations: assosiatedColorManipulations }) => {
					this.onColorInput({
						key: assosiatedColorKey,
						color: assosiatedColorValue || color,
						manipulations: assosiatedColorManipulations,
					});
				});
			}
		},

		parseColor({ color, format = 'hsl', manipulations = [] }) {
			const colorIsString = typeof color === 'string';
			let colorVal = colorIsString ? color.trim() : color;

			if (colorIsString) {
				// if it's css variable
				if (/^--[\w-]+$/.test(color)) {
					colorVal = this.rootStyles.getPropertyValue(color).trim();
				}
				// Has spaces in the middle, wrap with hsl()
				if (/\S+\s+\S+/.test(colorVal) && !colorVal.startsWith('hsl')) {
					colorVal = `hsl(${colorVal})`;
				}
			}

			let clr = tinycolor(colorVal);

			if ( manipulations.length ) {
				manipulations.forEach(({ type, value, condition }) => {
					if ( (condition === 'lightMode' && this.darkMode) || (condition === 'darkMode' && !this.darkMode) ) return;

					if ( type === 'autoBlackWhite' ) {
						return clr = clr.getLuminance() >= 0.4 ? tinycolor('#000') : tinycolor('#fff');
					}

					clr = clr[type](value);
				});
			}

			if ( Array.isArray(format) ) {
				const obj = {};

				format.forEach(format => {
					obj[format] = this.getColorString(clr, format);
				});

				return obj;
			}

			return this.getColorString(clr, format);
		},

		getColorString(colorObj, format = 'hsl') {
			if (format === 'colorObj') {
				return colorObj;
			}

			if (format === 'twHsl') {
				const hsl = colorObj.toHsl();

				return `${Math.ceil(hsl.h)} ${Math.ceil(hsl.s * 100)}% ${Math.ceil(hsl.l * 100)}%${hsl.a && hsl.a !== 1 ? ` / ${Math.ceil(hsl.a * 100)}%` : ''}`;
			}

			return colorObj.toString(format);
		},

		randomColors() {
			const randomColor = tinycolor.random();
			let sRange = [ 0.25, 1 ];
			let lRange = [ 1, 0.05 ];

			if (this.darkMode) {
				sRange[1] = 0.7;
				lRange = lRange.reverse();
			}

			const colorValues = generateColorRamp({
				total: 7,
				hStart: randomColor.toHsl().h,
				hStartCenter: 0,
				hCycles: 0.75,
				hEasing: (x, fr) => x + (-fr + Math.random() * fr * 2) * .5,

				sRange,
				lRange,

				sEasing: (x, fr) => x + (-fr + Math.random() * fr * 2) * .5,
				lEasing: (x, fr) => x + (-fr + Math.random() * fr * 2) * .5,
			});

			const navbarColorValues = generateColorRampWithCurve({
				hueList: colorUtils.colorHarmonies.splitComplementary(randomColor.toHsl().h),
				sRange: sRange.reverse(),
				lRange: lRange.reverse(),
				curveMethod: 'arc',
				curveAccent: 0.01,
			});

			const gradientColorValues = generateColorRamp({
				hueList: colorUtils.uniqueRandomHues({
					startHue: 0.00,
					total: 4,
					minDistance: 90,
				}),
				sRange: [ 0.5, 0.9 ],
				lRange: [ 0.5, 0.9 ],
			});


			const middleValue = Math.floor(colorValues.length / 2);
			const primary = colorValues[middleValue];
			const secondary = colorValues[middleValue - 1];
			const accent = colorValues[middleValue + 1];
			const background = colorValues[0];
			const foreground = colorValues[colorValues.length - 1];
			const border = colorValues[1];

			const navbarMiddleValue = Math.floor(navbarColorValues.length / 2);
			const navbarBackground = navbarColorValues[navbarColorValues.length - 1];
			const navbarForeground = navbarColorValues[0];
			const navbarActiveColor = navbarColorValues[navbarMiddleValue];

			const gradientFromValue = gradientColorValues[0];
			const gradientViaValue = gradientColorValues[1];
			const gradientToValue = gradientColorValues[2];

			const primaryColor = tinycolor({ h: primary[0], s: primary[1], l: primary[2] });
			const secondaryColor = tinycolor({ h: secondary[0], s: Math.min(1, secondary[1]), l: Math.min(1, secondary[2]) });
			const accentColor = tinycolor({ h: accent[0], s: Math.min(1, accent[1]), l: Math.min(1, accent[2]) });
			const foregroundColor = tinycolor({ h: foreground[0], s: Math.min(1, foreground[1]), l: Math.min(1, foreground[2]) });
			const backgroundColor = tinycolor({ h: background[0], s: Math.min(1, background[1]), l: Math.min(1, background[2]) }).desaturate(this.darkMode ? 30 : 0);
			const borderColor = tinycolor({ h: border[0], s: Math.min(1, border[1]), l: Math.min(1, border[2]) }).desaturate(20).setAlpha(0.55);

			const navbarBackgroundColor = tinycolor({ h: navbarBackground[0], s: Math.min(1, navbarBackground[1]), l: Math.min(1, navbarBackground[2]) }).desaturate(this.darkMode ? 30 : 0);
			const navbarForegroundColor = tinycolor({ h: navbarForeground[0], s: Math.min(1, navbarForeground[1]), l: Math.min(1, navbarForeground[2]) });
			const navbarActiveColorValue = tinycolor({ h: navbarActiveColor[0], s: Math.min(1, navbarActiveColor[1]), l: Math.min(1, navbarActiveColor[2]) })[this.darkMode ? 'lighten' : 'darken'](30);

			const gradientFromColor = tinycolor({ h: gradientFromValue[0], s: Math.min(1, gradientFromValue[1]), l: Math.min(1, gradientFromValue[2]) });
			const gradientViaColor = tinycolor({ h: gradientViaValue[0], s: Math.min(1, gradientViaValue[1]), l: Math.min(1, gradientViaValue[2]) });
			const gradientToColor = tinycolor({ h: gradientToValue[0], s: Math.min(1, gradientToValue[1]), l: Math.min(1, gradientToValue[2]) });

			this.onColorInput({
				key: 'colorMainPrimary',
				color: primaryColor,
				assosiatedKeys: [
					{
						key: 'colorMainPrimaryForeground',
						manipulations: [ { type: 'autoBlackWhite' } ]
					}
				]
			});
			this.onColorInput({
				key: 'colorMainSecondary',
				color: secondaryColor,
				assosiatedKeys: [
					{
						key: 'colorMainSecondaryForeground',
						manipulations: [ { type: 'autoBlackWhite' } ]
					}
				]
			});
			this.onColorInput({
				key: 'colorMainAccent',
				color: accentColor,
				assosiatedKeys: [
					{
						key: 'colorMainAccentForeground',
						manipulations: [ { type: 'autoBlackWhite' } ]
					}
				]
			});
			this.onColorInput({
				key: 'colorMainForeground',
				color: foregroundColor,
				assosiatedKeys: [
					{
						key: 'colorMainHeadingForeground',
						manipulations: [
							{
								type: 'darken',
								value: 22,
								condition: 'lightMode'
							},
							{
								type: 'lighten',
								value: 30,
								condition: 'darkMode'
							},
							{
								type: 'desaturate',
								value: 7,
								condition: 'lightMode'
							},
							{
								type: 'desaturate',
								value: 30,
								condition: 'darkMode'
							}
						]
					}
				]
			});
			this.onColorInput({ key: 'colorMainBackground', color: backgroundColor });
			this.onColorInput({ key: 'colorMainBorder', color: borderColor });

			this.onColorInput({ key: 'colorNavbarBackground', color: navbarBackgroundColor });
			this.onColorInput({ key: 'colorNavbarForeground', color: navbarForegroundColor });
			this.onColorInput({
				key: 'colorNavbarActiveBackground',
				color: navbarActiveColorValue,
				assosiatedKeys: [
					{
						key: 'colorNavbarActiveForeground'
					},
					{
						key: 'colorNavbarHoverForeground'
					},
					{
						key: 'colorNavbarHoverBackground'
					}
				]
			});

			this.onColorInput({ key: 'colorMainGradientFrom', color: gradientFromColor });
			this.onColorInput({ key: 'colorMainGradientVia', color: gradientViaColor });
			this.onColorInput({ key: 'colorMainGradientTo', color: gradientToColor });
		},

		addToGoogleFontLoadQueue(font, isPreview = false) {
			if ( isPreview && this.lqdFontPreview.includes(font) ) {
				return;
			}

			if (!isPreview) {
				return this.loadGoogleFontFull(font);
			}

			this.loadingFontPreviewQueue.push(font);

			this.loadGoogleFontPreview(font);

			this.lqdFontPreview.push(font);
		},

		// Throttled function
		loadGoogleFontPreview() {
			if (!this.loadingFontPreviewQueue.length) return;

			const fontFamilyString = this.loadingFontPreviewQueue.map(font => `family=${font}`).join('&').replaceAll(' ', '+');
			const textString = encodeURIComponent(this.loadingFontPreviewQueue.join(''));
			const fontUrl = `https://fonts.googleapis.com/css2?${fontFamilyString}&display=swap&text=${textString}`;

			this.createLink(fontUrl);

			this.loadingFontPreviewQueue = [];
		},

		loadGoogleFontFull(font) {
			const fontUrl = `https://fonts.googleapis.com/css2?family=${font.replaceAll(/ /g, '+')}&display=swap`;

			this.createLink(fontUrl);
		},

		createLink(fontUrl) {
			const link = document.createElement('link');

			link.href = fontUrl;
			link.rel = 'stylesheet';
			link.type = 'text/css';

			document.head.appendChild(link);
		},

		onImportStyles(str) {
			if (!str) return;

			let val = str.trim();

			const propValPairs = {};
			// Match selector blocks with their content - now includes # for ID selectors
			const cssBlockRegex = /([\w\s\.\:\-\#]+)\s*\{([^}]*)\}/g;
			let match;

			while ((match = cssBlockRegex.exec(val)) !== null) {
				const selector = match[1].trim();
				const properties = match[2].trim().split(';').filter(Boolean);

				// Initialize array for this selector if it doesn't exist
				propValPairs[selector] = propValPairs[selector] || [];

				// Extract property-value pairs without selector
				for (const prop of properties) {
					const colonIndex = prop.indexOf(':');
					if (colonIndex > 0) {
						const property = prop.slice(0, colonIndex).trim();
						const value = prop.slice(colonIndex + 1).trim();
						if (property && value) {
							propValPairs[selector].push({ property, value });
						}
					}
				}
			}

			Object.entries(this.options).forEach(([ key, option ]) => {
				const cssVar = option.cssVar;

				Object.entries(this.cssSelectors).forEach(([ darkOrLight, cssSelector ]) => {
					const newCss = propValPairs[cssSelector]?.find(s => s.property === cssVar);

					if (!newCss) return;

					const sanitizedValue = this.sanitizeCssValue(newCss.value);

					if ( option.type === 'color' ) {
						const parsedColor = this.parseColor({ color: sanitizedValue, format: [ 'twHsl', 'hex' ] });

						this.options[key].values[darkOrLight].twHsl = parsedColor.twHsl;
						this.options[key].values[darkOrLight].hex = parsedColor.hex;
					} else if ( option.type === 'font' ) {
						const font = sanitizedValue.replace(/['"]/g, '');
						this.options[key].value = font;

						this.addToGoogleFontLoadQueue(font);
					} else {
						this.options[key].value = sanitizedValue;
					}
				});
			});
		},

		sanitizeCssValue(value) {
			if (!value) return '';

			value = value.replace(/javascript:/gi, '')
				.replace(/expression\(/gi, '')
				.replace(/eval\(/gi, '')
				.replace(/url\(/gi, '')
				.replace(/data:/gi, '');

			value = value.replace(/[^\w\s,.#:;%\/\-+()[\]@!~="'*]/g, '');

			const singleQuotes = (value.match(/'/g) || []).length;
			const doubleQuotes = (value.match(/"/g) || []).length;

			if (singleQuotes % 2 !== 0) {
				value = value.replace(/'/g, '');
			}

			if (doubleQuotes % 2 !== 0) {
				value = value.replace(/"/g, '');
			}

			return value;
		},

		resetStyles() {
			Object.entries(this.options).forEach(([ key, option ]) => {
				if (option.type === 'color') {
					this.options[key].values.light = {};
					this.options[key].values.dark = {};
				} else if (option.type === 'font') {
					this.options[key].value = '';
				} else {
					this.options[key].value = '';
				}
			});
		},

		readFromLocalStorage() {
			let data = localStorage.getItem(this.localStorageKey);

			if ( !data ) {
				data = this.styleTag.textContent;
			}

			return data;
		},

		writeToLocalStorage() {
			const data = this.styleString.replace(/[\n\t]/g, '');

			if ( !data ) {
				this.emptyLocalStorage();
				return;
			}

			localStorage.setItem(this.localStorageKey, data);
		},

		emptyLocalStorage() {
			localStorage.removeItem(this.localStorageKey);
		},

		saveAndClose() {
			this.setBackEnd(this.styleString ?? '');
		},
		discardChanges() {
			const currentEdits = this.currentEdits?.replace(/[\n\t]/g, '')?.trim() ?? '';

			!currentEdits.length && this.resetStyles();

			this.emptyLocalStorage();

			this.setBackEnd('', true);
		},

		async setBackEnd(data = '', clear=false) {
			const fonts = {};

			Object.entries(this.options).forEach(([ key, option ]) => {
				if (option.type === 'font') {
					fonts[key] = option.value;
				}
			});

			const res = await fetch('/dashboard/admin/live-customizer', {
				method: 'POST',
				headers: {
					'Accept': 'application/json',
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({
					style: data,
					fonts: fonts,
					clear: clear
				})
			});

			let dataResponse = await res.json();

			if (! res.ok) {
				toastr.error('Error uploading the customizer styles');

				return;
			}

			if (dataResponse.status === 'success') {
				toastr.success(dataResponse.message);

				setTimeout(() => {
					this.emptyLocalStorage();

					window.location.reload();
				}, 1000);

				return;
			}

			toastr.error(dataResponse.message ?? 'Error uploading the customizer styles');
		}
	});
}

export function lqdCustomizerFontPicker(key) {
	return ({
		dropdownOpen: false,
		selectedFont: '',
		searchString: '',

		init() {
			this.$watch('selectedFont', val => {
				this.options[key].value = val;
			});
		},

		onSearchInput() {
			const value = this.$event.target.value.trim();

			if (!value) {
				this.fontsPaginationCurrent = this.fontsPaginationLimit;

				this.showingGoogleFonts = this.googleFontsList.slice(0, this.fontsPaginationLimit);
				return;
			}

			this.showingGoogleFonts = this.googleFontsList.filter(font => font.toLowerCase().includes(value.toLowerCase()));
		},

		loadMoreFonts() {
			this.fontsPaginationCurrent += this.fontsPaginationLimit;

			this.showingGoogleFonts = this.googleFontsList.slice(0, this.fontsPaginationCurrent);
		}
	});
}
