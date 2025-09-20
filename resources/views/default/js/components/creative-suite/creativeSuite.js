import { Sortable } from 'sortablejs';
import { debounce, defer, difference, throttle } from 'lodash';

/**
 * @type {import('alpinejs').AlpineComponent}
 */
export default ({ assetsUrl = '' }) => {
	return ({
		stage: null,
		layer: null,
		selectionTransformer: null,
		stageInitiated: false,
		activePanel: null,
		showWelcomeScreen: true,
		layersSearchString: '',
		layersSortable: null,
		selectedGroupChildNodes: [],
		selectionColors: {
			fill: 'hsla(205,90%,48%,0.15)',
			stroke: 'hsl(205,90%,48%)',
		},
		fetchedAssets: [],
		prevViews: [],
		currentView: 'home',
		currentDocId: null,
		currentDocName: null,
		updatingName: null,
		aiTasksQueue: [],
		activeTool: null,
		activeFillTab: 'color',
		editingTextNode: null,
		editingTextarea: null,
		container: null,
		konvajsContent: null,
		containerDimensions: { x: 0, y: 0, width: 0, height: 0 },
		konvajsContentDimensions: { x: 0, y: 0, width: 0, height: 0 },
		selectedFont: '',
		fontsSearchString: '',
		newItems: [],
		lqdFontPreview: [],
		googleFontsList: [],
		showingGoogleFonts: [],
		fontsPaginationCurrent: 20,
		fontsPaginationLimit: 20,
		loadingFontPreviewQueue: [],
		adaptiveResize: true,
		prevStageWidth: null,
		prevStageHeight: null,
		busy: false,
		maxZoom: 200,
		minZoom: 20,
		zoomOffsetX: 0,
		zoomOffsetY: 0,
		zoomLevel: 100,
		isSpacePressed: false,
		isPanning: false,
		reachedMinZoom: false,
		reachedMaxZoom: false,
		activeTooltipDropdown: null,
		guidelineOffset: 5,
		placeholderImage: `${assetsUrl}/img/misc/placeholder-1.jpg`,
		templatesList: [],
		loadingTemplates: true,
		loadingTemplatesFailed: false,
		promptLibraryShow: false,
		promptFilter: 'all',
		searchPromptStr: '',
		prompt: '',
		history: [],
		historyPointer: -1,
		maxHistoryLength: 30,
		historyListeners: [
			'fill',
			// 'fillPatternImage',
			// 'fillPatternX',
			// 'fillPatternY',
			// 'fillPatternOffset',
			// 'fillPatternOffsetX',
			// 'fillPatternOffsetY',
			// 'fillPatternScale',
			// 'fillPatternScaleX',
			// 'fillPatternScaleY',
			// 'fillPatternRotation',
			// 'fillPatternRepeat',
			'fillLinearGradientStartPoint',
			'fillLinearGradientStartPointX',
			'fillLinearGradientStartPointY',
			'fillLinearGradientEndPoint',
			'fillLinearGradientEndPointX',
			'fillLinearGradientEndPointY',
			'fillLinearGradientColorStops',
			'fillRadialGradientStartPoint',
			'fillRadialGradientStartPointX',
			'fillRadialGradientStartPointY',
			'fillRadialGradientEndPoint',
			'fillRadialGradientEndPointX',
			'fillRadialGradientEndPointY',
			'fillRadialGradientStartRadius',
			'fillRadialGradientEndRadius',
			'fillRadialGradientColorStops',
			'stroke',
			'strokeWidth',
			'lineJoin',
			'lineCap',
			'shadowColor',
			'shadowBlur',
			'shadowOffset',
			'shadowOffsetX',
			'shadowOffsetY',
			'shadowOpacity',
			'shadowEnabled',
			'dash',
			'x',
			'y',
			'width',
			'height',
			'opacity',
			'scale',
			'scaleX',
			'scaleY',
			'rotation',
			'offset',
			'offsetX',
			'offsetY',
			'fontFamily',
			'fontSize',
			'fontStyle',
			'fontVariant',
			'textDecoration',
			'text',
			'align',
			'verticalAlign',
			'padding',
			'lineHeight',
			'wrap',
			'ellipsis',
			'image',
			'crop',
			'blurRadius',
			'skew',
			'skewX',
			'skewY',
			'sides',
			'numPoints',
			'innerRadius',
			'outerRadius',
			'angle',
		],

		// to be used in other components
		get creativeSuite() {
			return this.$data;
		},

		get nodes() {
			return this.layer?.getChildren(node => {
				const name = node.name();
				const type = node.getType();

				return (
					(
						type === 'Shape' ||
						type === 'Group'
					) &&
					![ 'lqd-cs-guide-line', 'lqd-cs-highlighter', 'lqd-cs-selection-transformer', 'lqd-cs-selection-rectangle' ].includes(name)
				);
			}) ?? [];
		},

		/**
		 * @returns {Array}
		 */
		get selectedNodes() {
			return this.selectionTransformer?.nodes() ?? [];
		},
		set selectedNodes(options = { nodes: [], ids: [], event: {} }) {
			let nodes = options.nodes || [];
			let ids = options.ids || [];
			let event = options.event || {};
			let metaPressed = event.metaKey || event.ctrlKey;
			let shiftPressed = event.shiftKey;

			if ( ids.length && !nodes.length ) {
				nodes = this.nodes.filter(node => ids.includes(node.id()));
			}

			if ( nodes === 'all' ) {
				nodes = this.nodes;
			}

			if (shiftPressed) {
				let newNodesSelection = [ ...this.selectedNodes ];

				nodes.forEach(node => {
					const nodeIndex = newNodesSelection.findIndex(n => n.id() === node.id());
					if ( nodeIndex !== -1 ) {
						newNodesSelection.splice(nodeIndex, 1);
					} else {
						newNodesSelection.push(node);
					}
				});

				nodes = newNodesSelection.filter(node => {
					const ancestors = node.findAncestors('Group');
					const ancestorsInSelection = ancestors.some(ancestor => ancestor.getChildren(child => child.id() === node.id()));

					return !ancestorsInSelection;
				});
			}

			if ( !metaPressed ) {
				nodes.forEach(node => {
					const ancestors = node.findAncestors('Group');
					const ancestorsLength = ancestors.length;

					ancestors.forEach((ancestor, index) => {
						// we want to select the top most ancestor
						if ( ancestorsLength > 1 && index === 0 ) {
							return;
						}

						if (
							!nodes.find(n => n.id() === ancestor.id()) &&
							!this.selectedGroupChildNodes.find(n => n.id() === node.id())
						) {
							nodes.push(ancestor);
							nodes = nodes.filter(n => n.id() !== node.id());
						}
					});
				});
			}

			this.selectedGroupChildNodes.forEach(node => {
				const ancestors = node.findAncestors('Group');
				const isAncestorSelected = ancestors.some(ancestor => nodes.some(n => n.id() === ancestor.id()));
				const allSelectedGroups = nodes.filter(n => n.getClassName() === 'Group');
				const isInSelectedGroups = allSelectedGroups.find(group => group.findOne(`#${node.id()}`));

				if (
					!nodes.find(n => n.id() === node.id()) ||
					isAncestorSelected ||
					isInSelectedGroups
				) {
					node.draggable(false);
					this.selectedGroupChildNodes = this.selectedGroupChildNodes.filter(n => n.id() !== node.id());
				}
			});

			nodes = nodes.filter(node => !node.getAttr('locked'));

			this.selectionTransformer.nodes(nodes);

			this.selectionTransformer.zIndex(this.nodes.length + 1);

			this.$dispatch('nodes-selected', {
				nodes: this.selectedNodes,
				nodesCount: this.selectedNodes.length,
				selectedNodesRect: this.selectedNodesRect
			});
		},

		get selectedNodesRect() {
			if ( !this.selectedNodes.length ) {
				return {
					x: 0,
					y: 0,
				};
			}

			return this.selectionTransformer.getClientRect();
		},

		get selectedNodesProps() {
			const selectedNodes = this.selectedNodes;

			if (!selectedNodes.length) return {};

			const firstNode = selectedNodes[0];
			const firstRect = selectedNodes.find(n => n.getClassName() === 'Rect');
			const firstImage = selectedNodes.find(n => n.getClassName() === 'Image');
			const firstPolygon = selectedNodes.find(n => n.getClassName() === 'RegularPolygon');
			const firstStar = selectedNodes.find(n => n.getClassName() === 'Star');
			const firstRing = selectedNodes.find(n => n.getClassName() === 'Ring');
			const firstArc = selectedNodes.find(n => n.getClassName() === 'Arc');
			const firstWedge = selectedNodes.find(n => n.getClassName() === 'Wedge');
			const firstText = selectedNodes.find(n => n.getClassName() === 'Text');
			const firstTextPath = selectedNodes.find(n => n.getClassName() === 'TextPath');

			const props = {
				x: firstNode.x(),
				y: firstNode.y(),
				rotation: firstNode.rotation(),
				width: firstNode.width(),
				height: firstNode.height(),
				opacity: firstNode.opacity(),
				fill: firstNode.fill(),
				fillPatternX: firstNode.fillPatternX() ?? 0,
				fillPatternY: firstNode.fillPatternY() ?? 0,
				fillPatternScaleX: firstNode.fillPatternScaleX() ?? 0,
				fillPatternScaleY: firstNode.fillPatternScaleY() ?? 0,
				fillPatternRepeat: firstNode.fillPatternRepeat(),
				cornerRadius: (firstRect?.cornerRadius() || firstImage?.cornerRadius()) ?? 0,
				sides: firstPolygon?.sides() ?? 0,
				numPoints: firstStar?.numPoints() ?? 0,
				innerRadius: (firstStar?.innerRadius() || firstRing?.innerRadius() || firstArc?.innerRadius()) ?? 0,
				outerRadius: (firstStar?.outerRadius() || firstRing?.outerRadius() || firstArc?.outerRadius()) ?? 0,
				angle: (firstArc?.angle() || firstWedge?.angle()) ?? 0,
				fontFamily: (firstText?.fontFamily() || firstTextPath?.fontFamily()) ?? '',
				fontSize: (firstText?.fontSize() || firstTextPath?.fontSize()) ?? 0,
				fontStyle: (firstText?.fontStyle() || firstTextPath?.fontStyle()) ?? 'normal 400',
				textDecoration: (firstText?.textDecoration() || firstTextPath?.textDecoration()) ?? '',
				lineHeight: firstText?.lineHeight(),
				letterSpacing: (firstText?.letterSpacing() || firstTextPath?.letterSpacing()) ?? 0,
				align: (firstText?.align() || firstTextPath?.align()) ?? '',
				stroke: firstNode.stroke() ?? '',
				strokeWidth: firstNode.strokeWidth() ?? 0,
				dash: firstNode.dash() ?? 0,
				blurRadius: firstNode.blurRadius() ?? null,
				skewX: firstNode.skewX() ?? null,
				skewY: firstNode.skewY() ?? null,
				globalCompositeOperation: firstNode.globalCompositeOperation() ?? null,
			};

			return Object.fromEntries(
				Object.entries(props).map(([ prop, value ]) => {
					const allNodesHaveSameValue = selectedNodes.every(node => {
						if (this.isPropertyNotApplicableToNodeType(prop, node.getClassName())) {
							return true;
						}
						return node[prop]() === value;
					});

					const displayValue = allNodesHaveSameValue
						? this.formatPropertyValue(prop, value)
						: magicai_localize.mixed;

					return [ prop, displayValue ];
				})
			);
		},
		set selectedNodesProps(options = {
			x: null,
			y: null,
			rotation: null,
			width: null,
			height: null,
			scaleX: null,
			scaleY: null,
			opacity: null,
			fill: null,
			fillPatternX: null,
			fillPatternY: null,
			fillPatternScaleX: null,
			fillPatternScaleY: null,
			fillPatternRepeat: null,
			cornerRadius: null,
			sides: null,
			numPoints: null,
			innerRadius: null,
			outerRadius: null,
			angle: null,
			fontFamily: null,
			fontSize: null,
			fontStyle: null,
			textDecoration: null,
			lineHeight: null,
			letterSpacing: null,
			align: null,
			stroke: null,
			strokeWidth: null,
			dash: null,
			blurRadius: null,
			skewX: null,
			skewY: null,
			globalCompositeOperation: null,
			useRelativeValue: false
		}) {
			const useRelativeValue = this.selectedNodes.length > 1 || options.useRelativeValue;
			const properties = [
				'x',
				'y',
				'rotation',
				'width',
				'height',
				'scaleX',
				'scaleY',
				'opacity',
				'fill',
				'fillPatternX',
				'fillPatternY',
				'fillPatternScaleX',
				'fillPatternScaleY',
				'fillPatternRepeat',
				'cornerRadius',
				'sides',
				'numPoints',
				'innerRadius',
				'outerRadius',
				'angle',
				'fontFamily',
				'fontSize',
				'fontStyle',
				'textDecoration',
				'lineHeight',
				'letterSpacing',
				'align',
				'stroke',
				'strokeWidth',
				'dash',
				'blurRadius',
				'skewX',
				'skewY',
				'globalCompositeOperation',
			];

			this.selectedNodes.forEach(node => {
				const type = node.getClassName();

				properties.forEach(prop => {
					if (
						options[prop] == null ||
						this.isPropertyNotApplicableToNodeType(prop, type)
					) {
						return;
					}

					const currentPropVal = node[prop === 'fontWeight' ? 'fontStyle' : prop]();
					const currentPropValIsNumber = !isNaN(parseInt(currentPropVal));
					const value = currentPropValIsNumber && options[prop] === 'flip' ? (currentPropVal * -1) : options[prop];
					const parsedVal = currentPropValIsNumber ? parseFloat(value) : value;
					let newVal = currentPropValIsNumber && useRelativeValue ? (currentPropVal + parsedVal) : parsedVal;

					if ( prop === 'fontStyle' ) {
						const fontStyleArray = (currentPropVal ?? '').split(' ');
						let fontStyle = fontStyleArray[0] ?? 'normal';
						let fontWeight = fontStyleArray[1] ?? '400';

						if ( !isNaN(parseInt(options[prop])) ) {
							fontWeight = options[prop];
						} else {
							fontStyle = options[prop];
						}

						newVal = `${fontStyle} ${fontWeight}`;
					}

					if (prop === 'rotation' && ( [ 'Rect', 'Text', 'TextPath', 'Image' ].includes(type))) {
						let normalizedValue = ((newVal % 360) + 360) % 360;
						if (normalizedValue > 180) normalizedValue -= 360;
						newVal = normalizedValue;

						const { x, y, rotation } = this.rotateNodeAroundCenter(node, newVal);

						node.rotation(rotation);
						node.x(x);
						node.y(y);

						return;
					}

					switch (prop) {
						case 'width':
						case 'height':
						case 'cornerRadius':
						case 'innerRadius':
						case 'outerRadius':
							newVal = Math.max(0, newVal);
							break;
						case 'sides':
						case 'numPoints':
							newVal = Math.max(3, newVal);
							break;
						case 'angle':
							newVal = Math.max(0, Math.min(360, newVal));
							break;
						case 'opacity':
							newVal = Math.max(0, Math.min(1, newVal));
							break;
					}

					// this.applyFilters(node, options);

					node[prop](newVal);
				});
			});
		},

		init() {
			this.googleFontsList = window.lqdGoogleFontsList;

			this.beforeUnloadHandler = this.beforeUnloadHandler.bind(this);
			this.positionTooltip = this.positionTooltip.bind(this);
			this.handleEditinTextareaOutsideClick = this.handleEditinTextareaOutsideClick.bind(this);
			this.removeEditingTextarea = this.removeEditingTextarea.bind(this);
			this.setEditingTextareaTextAndRemove = this.setEditingTextareaTextAndRemove.bind(this);

			this.addToHistory = debounce(this.addToHistory.bind(this), 250, { leading: false, trailing: true } );
			this.loadGoogleFontPreview = throttle(this.loadGoogleFontPreview.bind(this), 300, { leading: false } );
			this.resizeStage = throttle(this.resizeStage.bind(this), 50);
			this.onWindowResize = throttle(this.onWindowResize.bind(this), 250);

			this.showingGoogleFonts = this.googleFontsList.slice(0, this.fontsPaginationLimit);

			this.$watch('selectedFont', font => {
				this.$dispatch('font-selected', { font });
			});

			this.$watch('nodes.length', length => {
				if ( this.beforeUnloadListened ) return;

				if (length > 0) {
					this.addBeforeUnloadListener();
				} else {
					this.removeBeforeUnloadListener();
				}
			});

			this.$watch('[zoomLevel, zoomOffsetX, zoomOffsetY]', () => {
				this.positionTooltip();
				this.positionEditingTextarea();
				this.updateSelectionTransformerStroke();
			});

			this.$watch('currentView', view => {
				document.body.classList.toggle('overflow-hidden', [ 'editor', 'gallery' ].includes(view));
			});

			this.initiateStage();
		},

		initiateStage(options = { width: 720, height: 720 }) {
			const isMobile = this.isMobile();
			const width = options.width ?? 720;
			const height = options.height ?? 720;

			if ( this.$refs.editorCanvasContainer === null ) {
				console.error('Creative Suite canvas was not found');
				return;
			}

			if ( this.stageInitiated || this.stage ) {
				console.warn('Creative Suite canvas was already initiated');
				return;
			}

			this.stage = new Konva.Stage({
				container: this.$refs.editorCanvasContainer,
				width,
				height,
				name: 'lqd-cs-stage',
				draggable: false
			});

			this.selectionTransformer = new Konva.Transformer({
				keepRatio: false,
				borderStroke: this.selectionColors.stroke,
				anchorSize: isMobile ? 10 : 6,
				rotationSnaps: [ 0, 45, 90, 135, 180, 225, 270, 315, 360 ],
				rotationSnapTolerance: 5,
				rotateAnchorOffset: isMobile ? 50 : 30,
				ignoreStroke: true,
				flipEnabled: false,
				id: 'lqd-cs-selection-transformer',
				name: 'lqd-cs-selection-transformer',
			});

			this.selectionRectangle = new Konva.Rect({
				fill: this.selectionColors.fill,
				stroke: this.selectionColors.stroke,
				strokeWidth: 1,
				visible: false,
				id: 'lqd-cs-selection-rectangle',
				name: 'lqd-cs-selection-rectangle',
			});

			this.layer = new Konva.Layer({
				name: 'lqd-cs-layer-main'
			});

			this.layer.add(this.selectionTransformer);
			this.layer.add(this.selectionRectangle);

			this.stage.add(this.layer);

			this.container = this.stage.container();

			this.konvajsContent = this.container.querySelector('.konvajs-content');

			this.container.tabIndex = 1;
			this.container.focus();

			this.setContainerDimensions();

			this.selectionTransformerEvents();

			this.handleSelectionRectangle();

			this.initLayers();

			this.stageEvents();

			this.containerEvents();

			this.handlePanAndZoom();

			this.initNodesSnap();

			this.positionTooltip();

			this.handleWindowResize();

			this.stageInitiated = true;
		},

		setContainerDimensions() {
			this.containerDimensions = {
				x: this.container.offsetLeft,
				y: this.container.offsetTop,
				width: this.container.clientWidth,
				height: this.container.offsetHeight,
			};

			this.konvajsContentDimensions = {
				x: this.konvajsContent.offsetLeft,
				y: this.konvajsContent.offsetTop,
				width: this.konvajsContent.clientWidth,
				height: this.konvajsContent.offsetHeight,
			};
		},

		handleSelectionRectangle() {
			let x1, y1, x2, y2;
			let childsCount = this.layer.getChildren().length;

			function onMouseUp(event) {
				// do nothing if we didn't start selection
				if (!this.selectionRectangle.visible() || this.isPanning || this.isSpacePressed) {
					return;
				}

				// update visibility in timeout, so we can check it in click event
				defer(() => {
					this.selectionRectangle.visible(false);
				});

				var box = this.selectionRectangle.getClientRect();
				var selectedNodes = this.nodes.filter(n => Konva.Util.haveIntersection(box, n.getClientRect()));

				this.selectedNodes = { nodes: selectedNodes, event };
			}

			this.stage.on('mousedown', event => {
				if ( this.isPanning || this.isSpacePressed || event.touches?.length >= 2 ) return;

				const node = event.target;

				if (node._id === this.stage._id) {
					childsCount = this.layer.getChildren().length;

					x1 = this.stage.getPointerPosition().x;
					y1 = this.stage.getPointerPosition().y;
					x2 = this.stage.getPointerPosition().x;
					y2 = this.stage.getPointerPosition().y;

					this.selectionRectangle.setAttrs({
						x: x1,
						y: y1,
						width: 0,
						height: 0,
						visible: true,
						zIndex: childsCount - 1
					});
				} else if ( this.nodes.find(n => n.id() === node.id()) && !this.selectedNodes.find(n => n.id() === node.id()) ) {
					this.selectedNodes = { nodes: [ node ], event: event.evt };
				}

			});

			this.stage.on('mousemove', event => {
				if (!this.selectionRectangle.visible() || this.isPanning || this.isSpacePressed || event.touches?.length >= 2) {
					return;
				}

				x2 = this.stage.getPointerPosition().x;
				y2 = this.stage.getPointerPosition().y;

				this.selectionRectangle.setAttrs({
					x: Math.min(x1, x2),
					y: Math.min(y1, y2),
					width: Math.abs(x2 - x1),
					height: Math.abs(y2 - y1),
					zIndex: childsCount - 1
				});
			});

			window.addEventListener('mouseup', onMouseUp.bind(this));

			this.stage.on('dblclick dbltap', event => {
				let target = event.target;

				if ( target._id === this.stage._id ) {
					return;
				}

				const ancestors = target.findAncestors('Group');

				if ( ancestors.length ) {
					if ( ancestors.length > 1 ) {
						target = ancestors[0];
					}

					target.draggable(true);
					this.selectedGroupChildNodes.push(target);
					this.selectedNodes = { nodes: [ target ], event: event.evt };
				}
			});

			this.stage.on('click tap', event => {
				const node = event.target;
				const shiftKey = event.evt.shiftKey;

				if (
					shiftKey ||
					(
						this.selectionRectangle.visible() &&
						this.selectionRectangle.width() > 0 &&
						this.selectionRectangle.height() > 0
					)
				) {
					return;
				}

				if (node._id === this.stage._id) {
					if ( this.selectedNodes.length ) {
						this.selectedNodes = {};
					}

					return;
				}

				this.selectedNodes = { nodes: [ node ], event: event.evt };
			});

			this.container.addEventListener('keydown', event => {
				const key = event.key;

				if ( !key || this.editingTextNode ) return;

				const metaPressed = event.ctrlKey || event.metaKey;
				const shiftPressed = event.shiftKey;
				let delta = 1;

				if ( metaPressed ) {
					delta = 0.1;
				}

				if ( shiftPressed ) {
					delta = 10;
				}

				switch (key) {
					case 'Backspace':
						event.preventDefault();
						this.destroySelectedNodes();
						break;
					case 'ArrowUp':
						event.preventDefault();
						this.selectedNodesProps = { y: -delta, useRelativeValue: true };
						this.positionTooltip();
						break;
					case 'ArrowDown':
						event.preventDefault();
						this.selectedNodesProps = { y: delta, useRelativeValue: true };
						this.positionTooltip();
						break;
					case 'ArrowLeft':
						event.preventDefault();
						this.selectedNodesProps = { x: -delta, useRelativeValue: true };
						this.positionTooltip();
						break;
					case 'ArrowRight':
						event.preventDefault();
						this.selectedNodesProps = { x: delta, useRelativeValue: true };
						this.positionTooltip();
						break;
					case 'a':
						event.preventDefault();
						metaPressed && (this.selectedNodes = { nodes: 'all' });
						break;
				}
			});

			this.konvajsContent.addEventListener('dragover', event => {
				event.preventDefault(); // important - must prevent default behavior
			});

			this.konvajsContent.addEventListener('drop', event => {
				event.preventDefault();

				this.stage.setPointersPositions(event);

				const data = JSON.parse(event.dataTransfer.getData('data') ?? '{}');

				if ( !data.type ) {
					return;
				}

				let { x, y } = this.stage.getPointerPosition();

				const node = this.addNodeToStage({
					type: data.type,
					attrs: {
						...data.attrs
					},
					beforeAdd: node => {
						const type = node.getClassName();
						const changePosition = type === 'Rect' || type === 'Path';
						const nodeRect = node.getClientRect();

						node.setAttrs({
							x: x - (changePosition ? nodeRect.width / 2 : 0),
							y: y - (changePosition ? nodeRect.height / 2 : 0),
						});
					}
				});

				this.selectedNodes = { nodes: [ node ] };
			});
		},

		stageEvents() {
			this.stage.on('click tap', () => {
				this.setActiveTooltipDropdown(null);
			});

			this.stage.on('dragmove', event => {
				const node = event.target;
				const nodeName = node.name();

				if ( nodeName === 'lqd-cs-stage' ) {
					node.x(0);
					node.y(0);
					return;
				}
			});

			this.stage.on('widthChange heightChange', this.addToHistory);
		},

		containerEvents() {
			const onClick = event => {
				if (event.target !== this.container) {
					return;
				}

				if ( this.selectedNodes.length ) {
					this.selectedNodes = {};
				}
			};

			this.container.addEventListener('click', onClick.bind(this));
			this.container.addEventListener('tap', onClick.bind(this));

			this.container.addEventListener('keydown', e => {
				// Ctrl/Cmd + Z for undo
				if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
					e.preventDefault();
					this.undo();
				}
				// Ctrl/Cmd + Shift + Z or Ctrl/Cmd + Y for redo
				if (((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'z') ||
					((e.ctrlKey || e.metaKey) && e.key === 'y')) {
					e.preventDefault();
					this.redo();
				}
			});
		},

		selectionTransformerEvents() {
			function applyScaleToShape(shape, scaleX, scaleY) {
				const type = shape.getClassName();

				if ( [ 'Rect', 'Image', 'Text' ].includes(type) ) {
					shape.width(shape.width() * scaleX);
					shape.height(shape.height() * scaleY);
					shape.scaleX(1);
					shape.scaleY(1);
				} else if (type === 'Circle' || type === 'RegularPolygon') {
					const originalRadius = shape.radius();
					const uniformScale = (scaleX + scaleY) / 2;
					shape.radius(originalRadius * uniformScale);
					shape.scaleX(1);
					shape.scaleY(1);
				} else if (type === 'Ellipse') {
					shape.radiusX(shape.radiusX() * scaleX);
					shape.radiusY(shape.radiusY() * scaleY);
					shape.scaleX(1);
					shape.scaleY(1);
				}
			}

			function applyScaleToChildren(parent, parentScaleX, parentScaleY) {
				parent.getChildren().forEach(child => {
					const childType = child.getClassName();

					const { x: relX, y: relY } = child.position();
					child.position({
						x: relX * parentScaleX,
						y: relY * parentScaleY
					});

					if (childType === 'Group') {
						const childScaleX = child.scaleX() * parentScaleX;
						const childScaleY = child.scaleY() * parentScaleY;
						child.scaleX(1);
						child.scaleY(1);
						applyScaleToChildren(child, childScaleX, childScaleY);
					} else {
						applyScaleToShape(child, parentScaleX * child.scaleX(), parentScaleY * child.scaleY());
					}
				});
			}

			this.selectionTransformer.on('transform', () => {
				const nodes = this.selectedNodes;

				nodes.forEach(node => {
					const type = node.getClassName();
					const scaleX = node.scaleX();
					const scaleY = node.scaleY();

					if (type === 'Group') {
						node.scaleX(1);
						node.scaleY(1);
						applyScaleToChildren(node, scaleX, scaleY);
						return;
					}

					applyScaleToShape(node, scaleX, scaleY);
				});
			});
		},

		createNode(options = { type: null, attrs: {} }) {
			if ( !this.stageInitiated ) return;

			if ( !Konva[options.type] ) {
				console.error(`Konva does not support the type "${options.type}"`);
				return;
			}

			const name = this.generateNameByType(options.type);
			// const shapesWithStroke = [ 'Line', 'Arrow' ];
			const stageWidth = this.stage.width();
			const stageHeight = this.stage.height();
			const attrs = {
				x: stageWidth / 2,
				y: stageHeight / 2,
				fill: Konva.Util.getRandomColor(),
				stroke: '',
				strokeWidth: 0,
				// pointerLength: 20,
				// pointerWidth: 20,
				draggable: true,
				strokeScaleEnabled: false,
				text: options.attrs?.text || magicai_localize.double_click_to_edit,
				fontSize: 30,
				fontStyle: 'normal 400',
				id: `${name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`,
				name,
				perfectDrawEnabled: false,
				offset: {
					x: 0,
					y: 0,
				},
				...options.attrs,
			};

			const node = new Konva[options.type](attrs);

			if ( attrs.x === 'left' ) {
				node.x(0);
			}
			if ( attrs.x === 'center' ) {
				node.x((stageWidth - node.width()) / 2);
			}
			if ( attrs.x === 'right' ) {
				node.x(stageWidth - node.width());
			}

			if ( attrs.y === 'top' ) {
				node.y(0);
			}
			if ( attrs.y === 'middle' ) {
				node.y((stageHeight - node.height()) / 2);
			}
			if ( attrs.y === 'bottom' ) {
				node.y(stageHeight - node.height());
			}

			// this.handleHighlighterVisibility(node);

			if ( options.type === 'Group' ) {
				this.addGroupHighlighter(node);
			}

			if ( [ 'Text', 'TextPath' ].includes( options.type)  ) {
				this.handleTextEdit(node);
			}

			if ( options.type === 'Image' ) {
				if ( attrs.fillSource ) {
					if ( this.fetchedAssets.includes(attrs.fillSource) || attrs.fillSource === this.placeholderImage ) {
						this.setNodeFillPattern({ node, url: attrs.fillSource });
					} else {
						this.handleUploadImage(node, attrs.fillSource);
					}
				} else {
					this.setNodeFillPattern({ node, url: this.placeholderImage });
				}
			}

			node.on('heightChange widthChange innerRadiusChange outerRadiusChange radiusChange radiusXChange radiusYChange', () => {
				this.handleFillCover(node);
			});

			node.on(this.historyListeners.map(l => l += 'Change').join(' '), this.addToHistory);

			return node;
		},

		addNodeToStage(options = { type: 'Ellipse', attrs: {}, beforeAdd: null, node: null, addHistory: true }) {
			if ( !this.stageInitiated ) return;

			const node = options.node || this.createNode(options);
			const addHistory = options.addHistory ?? true;

			if ( options.beforeAdd && typeof options.beforeAdd === 'function' ) {
				options.beforeAdd.call(this, node);
			}

			this.showWelcomeScreen = false;

			this.layer.add(node);

			this.selectedNodes = { nodes: [ node ] };

			if ( addHistory ) {
				this.addToHistory();
			}

			return node;
		},

		addGroupHighlighter(node) {
			const transformer = new Konva.Transformer({
				id: `highlighter-${node.id()}`,
				name: 'lqd-cs-highlighter',
				enabledAnchors: [],
				rotateEnabled: false,
				borderStroke: this.selectionColors.stroke,
				draggable: false
			});

			node.setAttr('highlighter', transformer.id());

			this.layer.add(transformer);
		},

		handleHighlighterVisibility(node) {
			let currentStroke = '';
			let currentStrokeWidth = '';
			let nodeToHighlight = node;

			node.on('mouseover', event => {
				const metaPressed = event.evt.ctrlKey || event.evt.metaKey;
				const ancestors = node.findAncestors('Group');
				const lastAncestor = ancestors.at(-1);

				if ( !metaPressed && lastAncestor ) {
					nodeToHighlight = lastAncestor;
				}

				if ( nodeToHighlight.getType() === 'Group' && metaPressed ) return;

				const highlighter = this.layer.findOne(`#highlighter-${nodeToHighlight.id()}`);
				const nodeIsSelected = this.selectedNodes.find(n => n.id() === nodeToHighlight.id());

				currentStroke = node.stroke();
				currentStrokeWidth = node.strokeWidth();

				if ( !nodeIsSelected ) {
					if ( highlighter ) {
						highlighter.nodes([ nodeToHighlight ]);
					}

					nodeToHighlight.stroke(this.selectionColors.stroke);
					nodeToHighlight.strokeWidth(currentStrokeWidth ?? 2);
				}
			});

			node.on('mouseout', () => {
				const highlighterId = nodeToHighlight.getAttr('highlighter');

				if ( highlighterId ) {
					const highlighter = this.layer.findOne(`#${highlighterId}`);

					highlighter.nodes([]);
				}

				nodeToHighlight.stroke(currentStroke);
				nodeToHighlight.strokeWidth(currentStrokeWidth);

				nodeToHighlight = node;
			});
		},

		applyFilters(node, options) {
			if ( options.blurRadius ) {
				node.clearCache();
				node.cache();
				node.filters([ Konva.Filters.Blur ]);
			} else if ( !options.blurRadius && !node.blurRadius() && node.isCached() ) {
				node.clearCache();
			}
		},

		handleTextEdit(node) {
			node.off('dblclick.editText dbltap.editText');

			node.on('dblclick.editText dbltap.editText', () => {
				if ( node.getAttr('aiTaskInProgress') ) {
					return toastr.warn('An AI task is in progress. please wait until it\'s done.');
				}

				node.hide();
				this.selectionTransformer.hide();

				this.editingTextarea = document.createElement('textarea');
				this.container.insertAdjacentElement('afterend', this.editingTextarea);

				this.editingTextarea.classList.add(
					'absolute',
					'border-none',
					'p-0',
					'm-0',
					'overflow-hidden',
					'bg-transparent',
					'outline-none',
					'resize-none',
					'whitespace-pre',
					'origin-top-left',
					'placeholder:text-transparent'
				);

				this.editingTextNode = node;

				const nodeFontStyleArray = node.fontStyle().split(' ');
				const initialText = this.editingTextNode.text();
				let fontStyle = nodeFontStyleArray[0];
				let fontWeight = nodeFontStyleArray[1];

				if ( !isNaN(parseInt(fontStyle)) ) {
					fontStyle = null;
					fontWeight = fontStyle;
				}

				this.editingTextarea.style.fontFamily = node.fontFamily();
				this.editingTextarea.style.textAlign = node.align();
				this.editingTextarea.style.color = node.fill().toString();
				this.editingTextarea.style.transformOrigin = 'top left';
				this.editingTextarea.style.textDecoration = node.textDecoration();
				this.editingTextarea.value = node.text();
				this.editingTextarea.placeholder = this.editingTextarea.value;

				if ( fontStyle ) {
					this.editingTextarea.style.fontStyle = fontStyle;
				}
				if ( fontWeight ) {
					this.editingTextarea.style.fontWeight = fontWeight;
				}

				this.editingTextarea.focus();
				this.editingTextarea.select();

				this.positionEditingTextarea();

				this.editingTextarea.addEventListener('keydown', event => {
					if (!this.editingTextarea) return;

					if (event.key === 'Enter' && !event.shiftKey) {
						this.setEditingTextareaTextAndRemove();
						return;
					}

					if (event.key === 'Escape') {
						this.setEditingTextareaTextAndRemove(initialText);
						return;
					}

					this.editingTextarea.placeholder = this.editingTextarea.value;

					this.editingTextarea.style.height = 'auto';
					this.editingTextarea.style.height = (this.editingTextarea.scrollHeight + node.fontSize()) + 'px';
				});

				this.$nextTick(() => {
					this.editingTextarea.addEventListener('focusout', this.setEditingTextareaTextAndRemove);
					window.addEventListener('click', this.handleEditinTextareaOutsideClick);
					window.addEventListener('tap', this.handleEditinTextareaOutsideClick);
				});
			});

			node.on('fontFamilyChange fontSizeChange fontStyleChange textDecorationChange lineHeightChange letterSpacingChange', ({ newVal, type }) => {
				if (newVal == null || type == null || !this.editingTextarea) return;

				const prop = type.replace('Change', '');
				let cssVal = newVal;

				if ( prop === 'letterSpacing' || prop === 'fontSize') {
					cssVal += 'px';
				}

				this.editingTextarea.style[prop] = cssVal;
			});
		},

		handleEditinTextareaOutsideClick(event) {
			if (!this.editingTextNode) return;

			if (event.target !== this.editingTextarea) {
				this.setEditingTextareaTextAndRemove(this.editingTextarea.value);
			}
		},

		removeEditingTextarea() {
			if (!this.editingTextNode || !this.editingTextarea) return;

			this.editingTextarea.removeEventListener('focusout', this.setEditingTextareaTextAndRemove);

			this.editingTextarea?.remove();

			window.removeEventListener('click', this.handleEditinTextareaOutsideClick);
			window.removeEventListener('tap', this.handleEditinTextareaOutsideClick);

			this.editingTextNode.show();

			this.selectionTransformer.show();
			this.selectionTransformer.forceUpdate();

			this.editingTextNode = null;
			this.editingTextarea = null;
		},

		setEditingTextareaTextAndRemove(text) {
			if ( !this.editingTextarea || !this.editingTextNode ) return;

			if ( typeof text !== 'string' ) {
				text = this.editingTextarea?.value ?? '';
			}

			this.editingTextNode.text(text);
			this.removeEditingTextarea();
		},

		positionEditingTextarea() {
			if ( !this.editingTextNode ) return;

			const nodeRect = this.editingTextNode.getClientRect();
			const zoom = this.zoomLevel / 100;
			const areaPosition = {
				x: nodeRect.x * zoom,
				y: nodeRect.y * zoom,
			};
			const contentRect = this.konvajsContent.getBoundingClientRect();
			const rotation = this.editingTextNode.rotation();
			const type = this.editingTextNode.getClassName();
			const isTextPath = type === 'TextPath';
			// setting minimum of 16px to prevent ios devices change page zoom
			const minFontSize = this.isMobile() ? 16 : 10;

			if (isTextPath) {
				areaPosition.x += (nodeRect.width / 2) * zoom;
				areaPosition.y += (nodeRect.height / 2) * zoom;
			}

			this.editingTextarea.style.width = 'auto';
			this.editingTextarea.style.height = 'auto';
			this.editingTextarea.style.top = Math.max(0, contentRect.top + areaPosition.y) + 'px';
			this.editingTextarea.style.left = Math.max(0, contentRect.left + areaPosition.x) + 'px';
			this.editingTextarea.style.fontSize = (Math.max(minFontSize, this.editingTextNode.fontSize() * zoom)) + 'px';
			this.editingTextarea.style.letterSpacing = (this.editingTextNode.letterSpacing() * zoom) + 'px';
			this.editingTextarea.style.lineHeight = isTextPath ? '1.2' : this.editingTextNode.lineHeight().toString();

			this.$nextTick(() => {
				this.editingTextarea.style.width = ((isTextPath ? this.stage.width() : this.editingTextNode.width()) * zoom) + 'px';
				this.editingTextarea.style.height = (this.editingTextarea.scrollHeight + (this.editingTextNode.fontSize() * this.editingTextNode.lineHeight())) + 'px';
			});

			if (rotation && !isTextPath) {
				this.editingTextarea.style.transform = `rotate(${rotation}deg)`;
			}
		},

		updateSelectionTransformerStroke() {
			const baseStrokeWidth = 1;
			const zoomFactor = 100 / this.zoomLevel;
			const adjustedWidth = baseStrokeWidth * Math.min(3, Math.max(0.5, zoomFactor));
			const baseAnchorSize = 6;
			const adjustedAnchorSize = baseAnchorSize * Math.min(3, Math.max(0.5, zoomFactor));

			this.selectionTransformer.borderStrokeWidth(adjustedWidth);
			this.selectionTransformer.anchorSize(adjustedAnchorSize);
		},

		/**
		 * Retrieves nodes from the internal node collection based on provided options.
		 * @param {Object} options - The options for filtering nodes.
		 * @param {Array<string>} [options.ids=[]] - Array of node IDs to filter by.
		 * @param {Array<Object>|string} [options.nodes=[]] - Array of node objects or 'all' to return all nodes.
		 * @param {string} [options.id=null] - A single node ID to find.
		 * @param {Object} [options.node=null] - A node object to find in the collection.
		 * @returns {Array<Object>} - Array of matching node objects.
		 */
		getNodesFromOptions(options = { ids: [], nodes: [], id: null, node: null }) {
			let nodes = [];

			if ( options.nodes?.length ) {
				if ( options.nodes === 'all' ) {
					nodes = this.nodes;
				} else {
					nodes = options.nodes;
				}
			} else if ( options.ids?.length ) {
				nodes = this.nodes.filter(node => options.ids.includes(node.id()));
			} else if ( options.id ) {
				nodes = [ this.nodes.find(node => node.id() === options.id) ];
			} else if ( options.node ) {
				if ( this.nodes.find(node => node.id() === options.node?.id()) ) {
					nodes = [ options.node ];
				} else {
					console.warn('Could not find the node: ', options.node?.id() ?? 'Unknown node ID');
				}
			}

			return nodes;
		},

		deselectNodes(options) {
			const nodeIds = this.getNodesFromOptions(options).map(n => n.id());

			this.selectedNodes = { nodes: this.selectedNodes.filter(node => !nodeIds.includes(node.id())) };
		},

		removeNodes(options) {
			const nodes = this.getNodesFromOptions(options);

			this.deselectNodes({ nodes });

			nodes.forEach(node => node.remove());
		},

		destroyNodes(options) {
			const nodes = this.getNodesFromOptions(options);

			this.deselectNodes({ nodes });

			nodes.forEach(node => node.destroy());

			if ( !this.nodes.length ) {
				this.showWelcomeScreen = true;
			}
		},

		destroySelectedNodes() {
			this.destroyNodes({ nodes: this.selectedNodes });
		},

		cloneSelectedNodes() {
			const nodes = this.selectedNodes.map(node => {
				const name = this.generateNameByType(node.getClassName());

				return node.clone({
					x: node.x() + 20,
					y: node.y() + 20,
					name,
					id: `${name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`,
				});
			});

			nodes.filter(n => [ 'Text', 'TextPath' ].includes(n.getClassName()))
				.forEach(node => this.handleTextEdit(node));

			this.layer.add(...nodes);

			this.selectedNodes = { nodes };
		},

		groupSelectedNodes() {
			const selectedNodes = this.selectedNodes;

			if ( !selectedNodes.length ) {
				return;
			}

			const name = this.generateNameByType('Group');

			// const group = new Konva.Group({
			// 	x: 0,
			// 	y: 0,
			// 	draggable: true,
			// 	name,
			// 	id: `${name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}`,
			// });

			const group = this.addNodeToStage({ type: 'Group', attrs: {
				x: 0,
				y: 0,
				width: null,
				height: null,
				fill: null,
				draggable: true,
			} });

			selectedNodes.forEach(node => {
				group.add(node);
				node.draggable(false);
			});

			this.layer.add(group);

			this.addGroupHighlighter(group);

			this.selectedNodes = { nodes: [ group ] };
		},

		ungroupSelectedNodes() {
			const selectedNodes = this.selectedNodes;

			if ( !selectedNodes.length ) {
				return;
			}

			const groups = selectedNodes.filter(node => node.getType() === 'Group');

			groups.forEach(group => {
				const children = group.getChildren();
				const groupAncestors = group.findAncestors('Group');
				const firstGroupAncestor = groupAncestors?.at(0);
				const groupId = group.id();
				const highlighterId = group.getAttr('highlighter');
				const nodes = [];

				children.forEach(child => {
					const ancestors = child.findAncestors('Group');

					// do not add inner groups
					if (ancestors.length <= 1) {
						nodes.push(child);
					}
				});

				nodes.forEach(node => {
					const absPos = node.getAbsolutePosition(firstGroupAncestor ?? this.stage);

					const nodeRotation = node.rotation();

					const groupRotation = group.rotation();
					const totalRotation = (nodeRotation + groupRotation) % 360;

					node.moveTo(firstGroupAncestor ?? this.layer);

					node.position(absPos);
					node.rotation(totalRotation);
					node.draggable(!firstGroupAncestor);
				});

				this.destroyNodes({ ids: [ groupId, highlighterId ] });

				this.selectedNodes = { nodes };
			});
		},

		getNodesByType(type) {
			if ( !type ) {
				return [];
			}

			return this.nodes
				.filter(node => node.getClassName() === type);
		},

		generateNameByType(type) {
			if ( !type ) {
				return 'Node 0';
			}

			const nodes = this.getNodesByType(type);
			const nodesLength = nodes.length;
			const nodeClassName = nodes[0]?.getClassName() || type;

			return `${nodeClassName} ${nodesLength + 1}`;
		},

		moveNode(action) {
			this.selectedNodes.forEach(node => node[action]());
		},

		initLayers() {
			const isMac = navigator.userAgentData ?
				navigator.userAgentData.platform.includes('macOS') :
				navigator.platform.includes('Mac');
			const multiDragKey = isMac ? 'Meta' : 'Control';

			this.layersSortable = Sortable.create(this.$refs.layers, {
				multiDrag: true,
				selectedClass: 'selected',
				multiDragKey,
				animation: 150,
				draggable: '.lqd-cs-layer',
				filter: '.lqd-cs-layers-list-template',
				onSelect: event => {
					const target = event.originalEvent.target;
					const itemsIds = event.items.map(item => item.dataset.id);

					this.selectedNodes = { ids: itemsIds };

					if ( !this.elementIsFocusable(target) ) {
						this.$nextTick(() => {
							this.container.focus();
						});
					}
				},
				onEnd: () => {
					const nodesCopy = [ ...this.nodes ];
					const selectedNodes = this.selectedNodes;

					const sortedArray = this.layersSortable.toArray();

					nodesCopy.sort((a, b) => {
						const indexA = sortedArray.indexOf(a.id());
						const indexB = sortedArray.indexOf(b.id());

						return indexB - indexA;
					});

					defer(() => this.removeNodes({ nodes: 'all' }));

					defer(() => {
						this.layer.add(...nodesCopy);
						this.selectedNodes = { nodes: selectedNodes };
					});
				}
			});
		},

		alignSelectedNodes(dir) {
			const selectedNodes = this.selectedNodes;

			if (selectedNodes.length === 1) {
				const node = selectedNodes[0];
				const nodeRect = node.getClientRect();
				const stageWidth = this.stage.width();
				const stageHeight = this.stage.height();
				const nodeX = node.x();
				const nodeY = node.y();

				switch (dir) {
					case 'left':
						node.x(nodeX - nodeRect.x);
						break;
					case 'center':
						node.x(nodeX - nodeRect.x + (stageWidth - nodeRect.width) / 2);
						break;
					case 'right':
						node.x(nodeX - nodeRect.x + stageWidth - nodeRect.width);
						break;
					case 'top':
						node.y(nodeY - nodeRect.y);
						break;
					case 'middle':
						node.y(nodeY - nodeRect.y + (stageHeight - nodeRect.height) / 2);
						break;
					case 'bottom':
						node.y(nodeY - nodeRect.y + stageHeight - nodeRect.height);
						break;
				}

				return;
			}

			const bounds = selectedNodes.reduce((acc, node) => {
				const rect = node.getClientRect();
				return {
					left: Math.min(acc.left, rect.x),
					right: Math.max(acc.right, rect.x + rect.width),
					top: Math.min(acc.top, rect.y),
					bottom: Math.max(acc.bottom, rect.y + rect.height),
					centerX: (acc.left + acc.right) / 2,
					centerY: (acc.top + acc.bottom) / 2
				};
			}, {
				left: Infinity,
				right: -Infinity,
				top: Infinity,
				bottom: -Infinity,
				centerX: 0,
				centerY: 0
			});

			bounds.centerX = (bounds.left + bounds.right) / 2;
			bounds.centerY = (bounds.top + bounds.bottom) / 2;

			selectedNodes.forEach(node => {
				const rect = node.getClientRect();
				const nodeX = node.x();
				const nodeY = node.y();

				switch (dir) {
					case 'left':
						node.x(nodeX - (rect.x - bounds.left));
						break;
					case 'center':
						node.x(nodeX + (bounds.centerX - (rect.x + rect.width/2)));
						break;
					case 'right':
						node.x(nodeX + (bounds.right - (rect.x + rect.width)));
						break;
					case 'top':
						node.y(nodeY - (rect.y - bounds.top));
						break;
					case 'middle':
						node.y(nodeY + (bounds.centerY - (rect.y + rect.height/2)));
						break;
					case 'bottom':
						node.y(nodeY + (bounds.bottom - (rect.y + rect.height)));
						break;
				}
			});
		},

		rotatePoint({ x, y }, rad) {
			const rcos = Math.cos(rad);
			const rsin = Math.sin(rad);

			return { x: x * rcos - y * rsin, y: y * rcos + x * rsin };
		},
		rotateNodeAroundCenter(node, rotation) {
			const topLeft = { x: -node.width() / 2, y: -node.height() / 2 };
			const current = this.rotatePoint(topLeft, Konva.getAngle(node.rotation()));
			const rotated = this.rotatePoint(topLeft, Konva.getAngle(rotation));
			const dx = rotated.x - current.x;
			const dy = rotated.y - current.y;
			const x = node.x() + dx;
			const y = node.y() + dy;

			return { rotation, x, y, };
		},

		isPropertyNotApplicableToNodeType(property, nodeType) {
			return (
				(property === 'cornerRadius' && ![ 'Rect', 'Image' ].includes(nodeType)) ||
				(property === 'sides' && nodeType !== 'RegularPolygon') ||
				(property === 'numPoints' && nodeType !== 'Star') ||
				(property === 'innerRadius' && ![ 'Star', 'Ring', 'Arc' ].includes(nodeType)) ||
				(property === 'outerRadius' && ![ 'Star', 'Ring', 'Arc' ].includes(nodeType)) ||
				(property === 'angle' && ![ 'Arc', 'Wedge' ].includes(nodeType)) ||
				([ 'fontSize', 'fontFamily', 'fontSize', 'fontStyle', 'fontWeight', 'textDecoration', 'letterSpacing', 'align' ].includes(property) && ![ 'Text', 'TextPath' ].includes(nodeType)) ||
				(property === 'lineHeight' && nodeType !== 'Text')
			);
		},

		formatPropertyValue(property, value) {
			if (typeof value === 'number') {
				if (property === 'opacity') {
					return value.toFixed(2);
				} else if (Number.isInteger(value)) {
					return value;
				} else {
					return value.toFixed(1);
				}
			}

			if ( property === 'fontStyle' ) {
				if ( value === 'normal' ) {
					return 400;
				}
			}

			return value;
		},

		handleFillCover(node) {
			if ( !node.getAttr('fillCover') ) return;

			this.applyFillCover(node);
		},

		async handleUploadImage(node, imageSource) {
			if ( imageSource ) {
				return fetch(imageSource)
					.then(response => response.blob())
					.then(async blob => {
						const file = new File([ blob ], 'image', { type: blob.type });

						const uploadedImage = await this.uploadImage({ file, node });

						await this.setNodeFillPattern({ node, url: uploadedImage.url });
					})
					.catch(error => {
						toastr.error('Failed to load image');
						console.error('Error creating file from URL:', error);
					});
			}

			const fileInput = document.createElement('input');

			fileInput.type = 'file';
			fileInput.accept = 'image/*';
			fileInput.style.display = 'none';

			document.body.appendChild(fileInput);

			fileInput.addEventListener('cancel', async event => {
				const input = event.target;
				const file = input.files[0];

				if ( !file && !node.fillPatternImage() ) {
					this.destroyNodes({ nodes: [ node ] });
				}
			});

			fileInput.addEventListener('change', async event => {
				const uploadedImage = await this.uploadImage({ event, node });

				await this.setNodeFillPattern({ node, url: uploadedImage.url });

				this.activeFillTab = 'image';

				this.$nextTick(() => {
					document.body.removeChild(fileInput);
				});
			});

			fileInput.click();
		},

		async uploadImage({ node, event, file }) {
			if ( event && !file ) {
				const input = event.target;
				file = input.files[0];
			}

			if (!file || !node) return;

			if (!file.type.startsWith('image/')) {
				toastr.error('Please upload a valid image file');
				return;
			}

			const formData = new FormData();

			formData.append('image', file);

			const uploadingImageRes = await fetch('/dashboard/user/creative-suite/image/upload', {
				method: 'POST',
				body: formData,
				headers: {
					'Accept': 'application/json'
				}
			});
			const uploadingImageData = await uploadingImageRes.json();

			return uploadingImageData.data;
		},

		async setNodeFillPattern({ node, url, setSize = true }) {
			const imageObj = new Image();
			// const nodeType = node.getClassName();

			return new Promise((resolve, reject) => {
				imageObj.onload = () => {
					node.fillPatternImage(imageObj);
					node.fillPriority('pattern');

					node.setAttr('fillImageWidth', imageObj.naturalWidth);
					node.setAttr('fillImageHeight', imageObj.naturalHeight);
					node.setAttr('fillSource', imageObj.src);

					// messing up undo/redo
					// if ( nodeType === 'Image' && setSize ) {
					// const fillImageWidth = imageObj.naturalWidth ?? 0;
					// const fillImageHeight = imageObj.naturalHeight ?? 0;

					// const imageAspectRatio = fillImageWidth / fillImageHeight;

					// const width = Math.min(fillImageWidth, this.stage.width() / 1.5);
					// const height = Math.min(fillImageHeight, this.stage.height() / 1.5) / imageAspectRatio;

					// node.width(width);
					// node.height(height);

					// node.x((this.stage.width() - width) / 2);
					// node.y((this.stage.height() - height) / 2);
					// }

					node.setAttr('fillCover', true);

					this.applyFillCover(node);

					resolve(imageObj);

					if ( !this.fetchedAssets.includes(url) ) {
						this.fetchedAssets.push(url);
					}
				};

				imageObj.onerror = function(error) {
					reject(error);
				};

				imageObj.src = url;
			});
		},

		removeFillPattern() {
			const selectedNode = this.selectedNodes[0];

			selectedNode.fillPatternImage(null);
			selectedNode.fillPriority('color');

			if ( this.$refs.selectedNodeFillPatternInput ) {
				this.$refs.selectedNodeFillPatternInput.value = '';
			}

			selectedNode.setAttr('fillImageWidth', null);
			selectedNode.setAttr('fillImageHeight', null);
			selectedNode.setAttr('fillSource', null);
			selectedNode.setAttr('fillCover', null);
			selectedNode.setAttr('fillAlign', null);
		},

		getFillCoverProps(node) {
			const fillAlign = node.getAttr('fillAlign') ?? 'center-middle';
			const nodeType = node.getClassName();
			const clientRect = node.getClientRect();

			const imageWidth = node.getAttr('fillImageWidth') ?? clientRect.width;
			const imageHeight = node.getAttr('fillImageHeight') ?? clientRect.height;

			let width, height, usesClientRect = false;

			if (nodeType === 'Circle') {
				width = height = node.radius() * 2;
			} else if (nodeType === 'Ellipse') {
				width = node.radiusX() * 2;
				height = node.radiusY() * 2;
			} else if ([ 'Rect', 'Image', 'Text' ].includes(nodeType)) {
				width = node.width();
				height = node.height();
			} else {
				width = clientRect.width;
				height = clientRect.height;
				usesClientRect = true;
			}

			const imageRatio = imageWidth / imageHeight;
			const nodeRatio = width / height;

			let scaleX, scaleY;
			if (nodeRatio >= imageRatio) {
				scaleX = width / imageWidth;
				scaleY = width / imageRatio / imageHeight;
			} else {
				scaleX = height * imageRatio / imageWidth;
				scaleY = height / imageHeight;
			}

			const newWidth = imageWidth * scaleX;
			const newHeight = imageHeight * scaleY;

			let offsetX = 0, offsetY = 0;

			const hasCenterOrigin = usesClientRect || ![ 'Rect', 'Image', 'Text' ].includes(nodeType);

			if (hasCenterOrigin) {
				if (fillAlign.includes('left')) {
					offsetX = -width / 2;
				} else if (fillAlign.includes('center')) {
					offsetX = -newWidth / 2;
				} else if (fillAlign.includes('right')) {
					offsetX = -newWidth + width / 2;
				}

				if (fillAlign.includes('top')) {
					offsetY = -height / 2;
				} else if (fillAlign.includes('middle')) {
					offsetY = -newHeight / 2;
				} else if (fillAlign.includes('bottom')) {
					offsetY = -newHeight + height / 2;
				}
			} else {
				if (fillAlign.includes('left')) {
					offsetX = 0;
				} else if (fillAlign.includes('center')) {
					offsetX = (width - newWidth) / 2;
				} else if (fillAlign.includes('right')) {
					offsetX = width - newWidth;
				}

				if (fillAlign.includes('top')) {
					offsetY = 0;
				} else if (fillAlign.includes('middle')) {
					offsetY = (height - newHeight) / 2;
				} else if (fillAlign.includes('bottom')) {
					offsetY = height - newHeight;
				}
			}

			if (nodeType === 'Path' && usesClientRect) {
				offsetX += clientRect.x;
				offsetY += clientRect.y;
			}

			return { offsetX, offsetY, scaleX, scaleY };
		},

		applyFillCover(node) {
			node = node ?? this.selectedNodes[0];

			if ( !node ) return;

			node.setAttr('fillPatternRepeat', 'no-repeat');

			const crop = this.getFillCoverProps( node );

			node.setAttrs({
				fillPatternX: crop.offsetX,
				fillPatternY: crop.offsetY,
				fillPatternScaleX: crop.scaleX,
				fillPatternScaleY: crop.scaleY,
			});
		},

		onFontsSearchInput() {
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

			this.createFontLink(fontUrl);

			this.loadingFontPreviewQueue = [];
		},

		async loadGoogleFontFull(font) {
			const fontUrl = `https://fonts.googleapis.com/css2?family=${font.replaceAll(/ /g, '+')}:wght@400;500;600;700;800;900&display=swap`;

			this.createFontLink(fontUrl);

			document.fonts.ready.finally(() => {
				// const fontLoaded = document.fonts.check(`1em ${font}`);

				if (this.selectedNodes.find(n => [ 'Text', 'TextPath' ].includes(n.getClassName()))) {
					this.selectionTransformer.forceUpdate();
				}
			});
		},

		createFontLink(fontUrl) {
			const link = document.createElement('link');

			link.href = fontUrl;
			link.rel = 'stylesheet';
			link.type = 'text/css';

			document.head.appendChild(link);
		},

		handleStageResize({ width, height, preserveAspectRatio = false }) {
			if (width == null && height == null) return;

			const currentWidth = parseInt(this.stage.width(), 10);
			const currentHeight = parseInt(this.stage.height(), 10);
			let newWidth = width ?? currentWidth;
			let newHeight = height ?? currentHeight;

			if (preserveAspectRatio && currentWidth && currentHeight) {
				const aspectRatio = currentWidth / currentHeight;

				if (width && !height) {
					newHeight = newWidth / aspectRatio;
				} else if (!width && height) {
					newWidth = newHeight * aspectRatio;
				} else if (width && height) {
					const newAspectRatio = newWidth / newHeight;

					if (newAspectRatio > aspectRatio) {
						newWidth = newHeight * aspectRatio;
					} else if (newAspectRatio < aspectRatio) {
						newHeight = newWidth / aspectRatio;
					}
				}
			}

			this.prevStageWidth = currentWidth;
			this.prevStageHeight = currentHeight;

			this.resizeStage(newWidth, newHeight);
		},

		resizeStage(width, height) {
			this.stage.width(width);
			this.stage.height(height);

			this.container.style.width = width + 'px';
			this.container.style.height = height + 'px';

			const scaleX = width / (this.prevStageWidth ?? 1);
			const scaleY = height / (this.prevStageHeight ?? 1);

			if (this.adaptiveResize && this.prevStageWidth > 0 && this.prevStageHeight > 0) {
				const isUpsizing = width > this.prevStageWidth || height > this.prevStageHeight;

				const uniformScale = isUpsizing ? Math.max(scaleX, scaleY) : Math.min(scaleX, scaleY);

				this.nodes.forEach(node => {
					node.x(node.x() * scaleX);
					node.y(node.y() * scaleY);

					const nodeType = node.getClassName();

					if (node.width && node.height) {
						const originalWidth = node.width();
						const originalHeight = node.height();

						node.width(originalWidth * uniformScale);
						node.height(originalHeight * uniformScale);
					}

					if (nodeType === 'Circle') {
						node.radius(node.radius() * uniformScale);
					}

					if (nodeType === 'Ellipse') {
						const radiusXOriginal = node.radiusX();
						const radiusYOriginal = node.radiusY();

						node.radiusX(radiusXOriginal * uniformScale);
						node.radiusY(radiusYOriginal * uniformScale);
					}

					if (nodeType === 'RegularPolygon') {
						node.radius(node.radius() * uniformScale);
					}

					if (nodeType === 'Star') {
						node.innerRadius(node.innerRadius() * uniformScale);
						node.outerRadius(node.outerRadius() * uniformScale);
					}

					if (nodeType === 'Text' || nodeType === 'TextPath') {
						node.fontSize(node.fontSize() * uniformScale);
					}

					if (node.fillPatternImage && node.fillPatternScaleX && node.fillPatternScaleY) {
						node.fillPatternScaleX(node.fillPatternScaleX() * uniformScale);
						node.fillPatternScaleY(node.fillPatternScaleY() * uniformScale);
					}
				});

				defer(() => {
					this.selectionTransformer.forceUpdate();
					this.fitToScreen();
					this.setContainerDimensions();
				});
			}
		},

		switchView(view) {
			if ( this.editingTextNode || this.tool ) return;

			if (view === '<') {
				this.currentView = this.prevViews.pop() || 'home';
				return;
			}

			this.prevViews.push(this.currentView);
			this.currentView = view || 'home';
		},

		getZoomPadding() {
			const styles = getComputedStyle(this.$el);
			const padding = 15;
			const x = parseInt(styles.getPropertyValue('--sidebar-w') ?? 0, 10) + padding;
			const y = parseInt(styles.getPropertyValue('--header-h') ?? 0, 10) + padding;

			return { x, y };
		},
		getFitToScreenZoom() {
			const zoomPadding = this.getZoomPadding();
			const w = (this.container.clientWidth - zoomPadding.x) / (this.konvajsContent.clientWidth * (this.zoomLevel / 100));
			const h = (this.container.clientHeight - zoomPadding.y) / (this.konvajsContent.clientHeight * (this.zoomLevel / 100));
			const zoom = Math.min(w, h) * 100;
			const fitZoom = Math.max(this.minZoom, Math.min(this.maxZoom, zoom));

			return fitZoom;
		},
		setZoomLevel(level) {
			this.zoomLevel = Math.max(this.minZoom, Math.min(this.maxZoom, level));
			this.reachedMinZoom = this.zoomLevel === this.minZoom;
			this.reachedMaxZoom = this.zoomLevel >= this.maxZoom;
		},
		zoomIn() {
			this.setZoomLevel(this.zoomLevel + 10);
		},
		zoomOut() {
			this.setZoomLevel(this.zoomLevel - 10);
		},
		fitToScreen() {
			const fitZoom = this.getFitToScreenZoom();

			this.setZoomLevel(fitZoom);

			const zoomPadding = this.getZoomPadding();

			this.zoomOffsetX = zoomPadding.x / 2;
			this.zoomOffsetY = zoomPadding.y / 2;
		},

		positionTooltip() {
			if (!this.selectedNodes.length) return;

			const contentRect = this.konvajsContent.getBoundingClientRect();
			const rect = this.selectedNodesRect;
			const zoom = this.zoomLevel / 100;

			// Calculate initial position
			const rawX = contentRect.left + (rect.x * zoom) + (rect.width * zoom / 2);
			const rawY = contentRect.top + (rect.y * zoom) - 40;

			// Get tooltip dimensions
			const tooltipRect = this.$refs.tooltip.getBoundingClientRect();
			const tooltipHeight = tooltipRect.height;

			// Allow tooltip to go half outside of konvajsContent boundaries
			const x = Math.max(
				contentRect.left,
				Math.min(rawX, contentRect.right )
			);
			const y = Math.max(
				contentRect.top - tooltipHeight / 2,
				Math.min(rawY, contentRect.bottom - tooltipHeight / 2)
			);

			if (!this._tooltipAnimationFrame) {
				this._tooltipAnimationFrame = requestAnimationFrame(() => {
					this.$refs.tooltip.style.left = `${x}px`;
					this.$refs.tooltip.style.top = `${y}px`;
					this._tooltipAnimationFrame = null;
				});
			}
		},

		handlePanAndZoom() {
			let lastMousePosition = { x: 0, y: 0 };
			let lastTouchDistance = 0;
			let lastTouchCenter = { x: 0, y: 0 };
			let touchPanStartPosition = { x: 0, y: 0 };
			let touchPanStartOffset = { x: 0, y: 0 };

			const handleWheel = e => {
				if (this.showWelcomeScreen) return;

				e.preventDefault();

				const rect = this.konvajsContent.getBoundingClientRect();
				const mouseX = e.clientX - rect.left;
				const mouseY = e.clientY - rect.top;

				if (e.metaKey || e.ctrlKey) {
					const zoomDirection = e.deltaY < 0 ? 1 : -1;
					const zoomDelta = zoomDirection * 2;
					const newZoomLevel = Math.max(
						this.minZoom,
						Math.min(this.maxZoom, this.zoomLevel + zoomDelta)
					);

					const zoomFactor = newZoomLevel / this.zoomLevel;
					this.setZoomLevel(newZoomLevel);

					const viewportCenter = {
						x: rect.width / 2,
						y: rect.height / 2
					};

					const mouseOffset = {
						x: (mouseX - viewportCenter.x) / this.zoomLevel,
						y: (mouseY - viewportCenter.y) / this.zoomLevel
					};

					this.zoomOffsetX -= mouseOffset.x * (zoomFactor - 1);
					this.zoomOffsetY -= mouseOffset.y * (zoomFactor - 1);
				} else {
					const panSpeed = this.zoomLevel / 15;
					this.zoomOffsetX -= e.deltaX / panSpeed;
					this.zoomOffsetY -= e.deltaY / panSpeed;
				}
			};

			const handleKeyDown = e => {
				if ( this.docActiveElementIsWritable() ) return;

				if (e.code === 'Space' && !e.repeat && !this.isSpacePressed && !this.showWelcomeScreen) {
					this.isSpacePressed = true;
					document.body.classList.add('is-panning');
					this.container.classList.add('is-panning');

					if (document.activeElement && document.activeElement !== document.body) {
						document.activeElement.blur();
					}
					this.container.focus();
					e.preventDefault();
				}
			};

			const handleKeyUp = e => {
				if ( this.docActiveElementIsWritable() ) return;

				if (e.code === 'Space') {
					this.isSpacePressed = false;
					this.isPanning = false;
					document.body.classList.remove('is-panning', 'is-panning-active');
					this.container.classList.remove('is-panning', 'is-panning-active');
				}
			};

			const handleMouseDown = e => {
				if ( this.docActiveElementIsWritable() ) return;

				if (this.isSpacePressed && e.button === 0 && !this.showWelcomeScreen) {
					this.isPanning = true;
					lastMousePosition = { x: e.clientX, y: e.clientY };
					document.body.classList.add('is-panning-active');
					this.container.classList.add('is-panning-active');

					if (document.activeElement && document.activeElement !== document.body) {
						document.activeElement.blur();
					}
					this.container.focus();
					e.stopPropagation();
					e.preventDefault();
				}
			};

			const handleMouseMove = e => {
				if ( this.docActiveElementIsWritable() ) return;

				if (this.isPanning) {
					const dx = e.clientX - lastMousePosition.x;
					const dy = e.clientY - lastMousePosition.y;

					lastMousePosition = { x: e.clientX, y: e.clientY };

					this.zoomOffsetX += dx / (this.zoomLevel / 15);
					this.zoomOffsetY += dy / (this.zoomLevel / 15);

					e.preventDefault();
					e.stopPropagation();
				}
			};

			const handleMouseUp = () => {
				if ( this.docActiveElementIsWritable() ) return;

				if (this.isPanning) {
					this.isPanning = false;
					document.body.classList.remove('is-panning-active');
					this.container.classList.remove('is-panning-active');
				}
			};

			const handleMouseLeave = () => {
				if ( this.docActiveElementIsWritable() ) return;

				if (this.isPanning) {
					this.isPanning = false;
					document.body.classList.remove('is-panning-active');
					this.container.classList.remove('is-panning-active');
				}
			};

			const handleTouchStart = e => {
				const touches = e.touches;

				if (this.showWelcomeScreen || touches.length < 2) return;

				const touch1 = touches[0];
				const touch2 = touches[1];

				if ( touch1 && touch2 ) {
					this.stage.stopDrag(false);
				}

				lastTouchDistance = Math.hypot(
					touch2.clientX - touch1.clientX,
					touch2.clientY - touch1.clientY
				);

				const rect = this.konvajsContent.getBoundingClientRect();
				lastTouchCenter = {
					x: ((touch1.clientX + touch2.clientX) / 2) - rect.left,
					y: ((touch1.clientY + touch2.clientY) / 2) - rect.top
				};

				touchPanStartPosition = {
					x: (touch1.clientX + touch2.clientX) / 2,
					y: (touch1.clientY + touch2.clientY) / 2
				};
				touchPanStartOffset = { x: this.zoomOffsetX, y: this.zoomOffsetY };
			};

			const handleTouchMove = e => {
				const touches = e.touches;
				if (this.showWelcomeScreen || touches.length < 2) return;

				e.preventDefault();

				if (touches.length === 2) {
					const touch1 = touches[0];
					const touch2 = touches[1];

					if (touch1 && touch2) {
						const currentTouchCenter = {
							x: (touch1.clientX + touch2.clientX) / 2,
							y: (touch1.clientY + touch2.clientY) / 2
						};

						const dx = currentTouchCenter.x - touchPanStartPosition.x;
						const dy = currentTouchCenter.y - touchPanStartPosition.y;

						this.zoomOffsetX = touchPanStartOffset.x + dx / (this.zoomLevel / 15);
						this.zoomOffsetY = touchPanStartOffset.y + dy / (this.zoomLevel / 15);

						const currentDistance = Math.hypot(
							touch2.clientX - touch1.clientX,
							touch2.clientY - touch1.clientY
						);

						const scaleFactor = currentDistance / lastTouchDistance;

						if (lastTouchDistance > 0 && scaleFactor !== 1) {
							const rect = this.konvajsContent.getBoundingClientRect();
							const currentCenter = {
								x: ((touch1.clientX + touch2.clientX) / 2) - rect.left,
								y: ((touch1.clientY + touch2.clientY) / 2) - rect.top
							};

							const newZoomLevel = Math.max(this.minZoom, Math.min(
								this.maxZoom,
								this.zoomLevel * scaleFactor
							));

							const oldZoom = this.zoomLevel;
							const newZoom = newZoomLevel;
							const zoomFactor = newZoom / oldZoom;

							this.setZoomLevel(newZoomLevel);

							const viewportCenterX = rect.width / 2;
							const viewportCenterY = rect.height / 2;
							const touchOffsetX = (lastTouchCenter.x - viewportCenterX) / oldZoom;
							const touchOffsetY = (lastTouchCenter.y - viewportCenterY) / oldZoom;

							this.zoomOffsetX -= touchOffsetX * (zoomFactor - 1);
							this.zoomOffsetY -= touchOffsetY * (zoomFactor - 1);

							lastTouchDistance = currentDistance;
							lastTouchCenter = currentCenter;
						}
					}
				}
			};

			const handleTouchEnd = e => {
				if (this.showWelcomeScreen) return;

				lastTouchDistance = 0;
				lastTouchCenter = { x: 0, y: 0 };

				this.stage.draggable(true);
			};

			this.container.addEventListener('wheel', handleWheel, { passive: false });
			this.container.addEventListener('mousedown', handleMouseDown);
			this.container.addEventListener('touchstart', handleTouchStart);
			this.container.addEventListener('touchmove', handleTouchMove);
			this.container.addEventListener('touchend', handleTouchEnd);
			this.container.addEventListener('touchcancel', handleTouchEnd);

			document.addEventListener('keydown', handleKeyDown);
			document.addEventListener('keyup', handleKeyUp);
			document.addEventListener('mousemove', handleMouseMove);
			document.addEventListener('mouseup', handleMouseUp);
			document.addEventListener('mouseleave', handleMouseLeave);
			window.addEventListener('resize', this.positionTooltip);

			this.selectionTransformer.on('transform', this.positionTooltip);

			document.addEventListener('nodes-selected', this.positionTooltip);
		},

		initNodesSnap() {
			let nodes = this.nodes;
			// Store initial node offsets relative to transformer
			let nodeOffsets = [];

			// were can we snap our objects?
			const getLineGuideStops = () => {
				// we can snap to stage borders and the center of the stage
				var vertical = [ 0, this.stage.width() / 2, this.stage.width() ];
				var horizontal = [ 0, this.stage.height() / 2, this.stage.height() ];

				// and we snap over edges and center of each object on the canvas
				nodes.forEach(guideItem => {
					if (this.selectionTransformer.nodes().find(n => n.id() === guideItem.id())) {
						return;
					}
					var box = guideItem.getClientRect();

					// and we can snap to all edges of shapes
					vertical.push([ box.x, box.x + box.width, box.x + box.width / 2 ]);
					horizontal.push([ box.y, box.y + box.height, box.y + box.height / 2 ]);
				});

				return {
					vertical: vertical.flat(),
					horizontal: horizontal.flat(),
				};
			};

			// what points of the object will trigger to snapping?
			// it can be just center of the object
			// but we will enable all edges and center
			const getObjectSnappingEdges = () => {
				const box = this.selectionTransformer.getClientRect();
				const absPos = this.selectionTransformer.absolutePosition();

				box.height = box.height - this.selectionTransformer.rotateAnchorOffset();
				box.y = box.y + this.selectionTransformer.rotateAnchorOffset();

				return {
					vertical: [
						{
							guide: Math.round(box.x),
							offset: Math.round(absPos.x - box.x),
							snap: 'start',
						},
						{
							guide: Math.round(box.x + box.width / 2),
							offset: Math.round(absPos.x - box.x - box.width / 2),
							snap: 'center',
						},
						{
							guide: Math.round(box.x + box.width),
							offset: Math.round(absPos.x - box.x - box.width),
							snap: 'end',
						},
					],
					horizontal: [
						{
							guide: Math.round(box.y),
							offset: Math.round(absPos.y - box.y),
							snap: 'start',
						},
						{
							guide: Math.round(box.y + box.height / 2),
							offset: Math.round(absPos.y - box.y - box.height / 2),
							snap: 'center',
						},
						{
							guide: Math.round(box.y + box.height),
							offset: Math.round(absPos.y - box.y - box.height),
							snap: 'end',
						},
					],
				};
			};

			// find all snapping possibilities
			const getGuides = (lineGuideStops, itemBounds) => {
				var resultV = [];
				var resultH = [];

				lineGuideStops.vertical.forEach(lineGuide => {
					itemBounds.vertical.forEach(itemBound => {
						var diff = Math.abs(lineGuide - itemBound.guide);
						// if the distance between guild line and object snap point is close we can consider this for snapping
						if (diff < this.guidelineOffset) {
							resultV.push({
								lineGuide: lineGuide,
								diff: diff,
								snap: itemBound.snap,
								offset: itemBound.offset,
							});
						}
					});
				});

				lineGuideStops.horizontal.forEach(lineGuide => {
					itemBounds.horizontal.forEach(itemBound => {
						var diff = Math.abs(lineGuide - itemBound.guide);
						if (diff < this.guidelineOffset) {
							resultH.push({
								lineGuide: lineGuide,
								diff: diff,
								snap: itemBound.snap,
								offset: itemBound.offset,
							});
						}
					});
				});

				var guides = [];

				// find closest snap
				var minV = resultV.sort((a, b) => a.diff - b.diff)[0];
				var minH = resultH.sort((a, b) => a.diff - b.diff)[0];
				if (minV) {
					guides.push({
						lineGuide: minV.lineGuide,
						offset: minV.offset,
						orientation: 'V',
						snap: minV.snap,
					});
				}
				if (minH) {
					guides.push({
						lineGuide: minH.lineGuide,
						offset: minH.offset,
						orientation: 'H',
						snap: minH.snap,
					});
				}
				return guides;
			};

			const drawGuides = guides => {
				guides.forEach(lg => {
					const isHorizontal = lg.orientation === 'H';

					var line = new Konva.Line({
						points: isHorizontal ? [ -6000, 0, 6000, 0 ] : [ 0, -6000, 0, 6000 ],
						stroke: this.selectionColors.stroke,
						strokeWidth: 1,
						name: 'lqd-cs-guide-line',
						dash: [ 4, 6 ],
					});

					this.layer.add(line);

					line.absolutePosition({
						x: isHorizontal ? 0 : lg.lineGuide,
						y: isHorizontal ? lg.lineGuide : 0,
					});
				});
			};

			this.selectionTransformer.on('dragstart', () => {
				nodes = this.nodes;

				// Store initial offsets between each node and the transformer
				nodeOffsets = [];
				const transformerPos = this.selectionTransformer.absolutePosition();

				this.selectedNodes.forEach(node => {
					const nodePos = node.absolutePosition();
					nodeOffsets.push({
						id: node.id(),
						xDiff: nodePos.x - transformerPos.x,
						yDiff: nodePos.y - transformerPos.y
					});
				});
			});

			this.selectionTransformer.on('dragmove', e => {
				// clear all previous lines on the screen
				this.layer.find('.lqd-cs-guide-line').forEach(l => l.destroy());

				// find possible snapping lines
				var lineGuideStops = getLineGuideStops();
				// find snapping points of current object
				var itemBounds = getObjectSnappingEdges();

				// now find where can we snap current object
				var guides = getGuides(lineGuideStops, itemBounds);

				// do nothing of no snapping
				if (!guides.length) {
					// If there's no snapping, we still need to position the tooltip
					this.positionTooltip();
					return;
				}

				drawGuides(guides);

				// Get the new transformer position with snapping applied
				const transformerAbsPos = { ...this.selectionTransformer.absolutePosition() };

				// Apply snapping to transformer position
				guides.forEach(lg => {
					switch (lg.orientation) {
						case 'V': {
							transformerAbsPos.x = lg.lineGuide + lg.offset;
							break;
						}
						case 'H': {
							transformerAbsPos.y = lg.lineGuide + lg.offset;
							break;
						}
					}
				});

				// Update transformer position
				this.selectionTransformer.absolutePosition(transformerAbsPos);

				// Update all selected nodes using the stored offsets
				this.selectedNodes.forEach(node => {
					const offset = nodeOffsets.find(o => o.id === node.id());
					if (offset) {
						node.absolutePosition({
							x: transformerAbsPos.x + offset.xDiff,
							y: transformerAbsPos.y + offset.yDiff
						});
					}
				});

				// Position tooltip only after all nodes are updated
				// This ensures the tooltip position is based on the final node positions
				this.positionTooltip();
			});

			this.layer.on('dragend', () => {
				this.layer.find('.lqd-cs-guide-line').forEach(l => l.destroy());
			});
		},

		addToHistory() {
			if ( this.history.length > this.maxHistoryLength - 1 ) {
				this.history.shift();
			}

			if ( this.historyPointer !== this.history.length - 1  ) {
				this.history = this.history.slice(0, this.historyPointer + 1);
			}

			this.history.push({
				stage: {
					width: this.stage.width(),
					height: this.stage.height()
				},
				zoom: {
					zoomOffsetX: this.zoomOffsetX,
					zoomOffsetY: this.zoomOffsetY,
					zoomLevel: this.zoomLevel
				},
				nodes: this.nodes.map(node => node.toJSON()),
				selectedNodeIds: this.selectedNodes.map(n => n.id())
			});

			this.historyPointer = Math.min(this.history.length - 1, this.historyPointer + 1);
		},

		undo() {
			if ( this.historyPointer === -1 ) return;

			this.historyPointer = Math.max(-1, this.historyPointer - 1);

			this.applyHistory();
		},

		redo() {
			if ( this.historyPointer === this.history.length - 1 ) return;

			this.historyPointer = Math.min(this.history.length - 1, this.historyPointer + 1);

			this.applyHistory();
		},

		applyHistory() {
			if ( this.historyPointer === -1 ) {
				return this.destroyNodes({ nodes: 'all' });
			}

			const history = this.history[this.historyPointer];

			if ( !history ) return;

			const stageWidth = this.stage.width();
			const stageHeight = this.stage.height();
			let stageSizeChanged = false;

			if ( history.stage?.width && history.stage.width !== stageWidth) {
				this.stage.width(history.stage.width);
				stageSizeChanged = true;
			}
			if ( history.stage?.height && history.stage.height !== stageHeight) {
				this.stage.height(history.stage.height);
				stageSizeChanged = true;
			}

			if ( stageSizeChanged && history.zoom ) {
				this.$nextTick(() => {
					this.fitToScreen();
				});
			}

			this.destroyNodes({ nodes: 'all' });

			history.nodes?.forEach(nodeData => {
				try {
					const tempNode = Konva.Node.create(nodeData);
					const className = tempNode.getClassName();
					const attrs = tempNode.getAttrs();

					this.addNodeToStage({
						type: className,
						attrs: attrs,
						addHistory: false
					});

					tempNode.destroy();

					if ( history.selectedNodeIds ) {
						this.selectedNodes = { ids: history.selectedNodeIds };
					}
				} catch (nodeError) {
					console.error('Error creating node:', nodeError);
				}
			});
		},

		downloadImage() {
			if (!this.stage) return;

			this.busy = true;

			try {
				const transformerVisible = this.selectionTransformer.visible();
				this.selectionTransformer.visible(false);
				const container = document.createElement('div');
				container.style.position = 'absolute';
				container.style.zIndex = '-1';
				container.style.opacity = '0';
				container.style.width = `${this.stage.width()}px`;
				container.style.height = `${this.stage.height()}px`;
				document.body.appendChild(container);
				const tempStage = new Konva.Stage({
					container: container,
					width: this.stage.width(),
					height: this.stage.height(),
					background: 'white'
				});
				const clonedLayer = this.layer.clone();

				clonedLayer.find('#lqd-cs-selection-transformer, #lqd-cs-selection-rectangle').forEach(node => {
					node.remove();
				});

				tempStage.add(clonedLayer);

				tempStage.toDataURL({
					mimeType: 'image/png',
					quality: 1,
					pixelRatio: 2,
					callback: dataUrl => {
						const link = document.createElement('a');
						link.download = `canvas-export-${new Date().getTime()}.png`;
						link.href = dataUrl;
						document.body.appendChild(link);
						link.click();
						document.body.removeChild(link);

						document.body.removeChild(container);
						tempStage.destroy();
						this.selectionTransformer.visible(transformerVisible);
						this.busy = false;
					}
				});
			} catch (error) {
				console.error('Error exporting image:', error);
				this.busy = false;
				this.selectionTransformer.visible(true);
			}
		},

		resetCanvas() {
			if (
				this.nodes.length &&
				confirm(magicai_localize['all_your_current_edits_will_be_lost__are_you_sure_'])
			) {
				this.resetHistory();
				this.destroyNodes({ nodes: 'all' });
				this.fitToScreen();
				this.currentDocId = null;
				this.currentDocName = null;
			}
		},

		resetHistory() {
			this.history = [];
			this.historyPointer = -1;
		},

		beforeUnloadHandler(event) {
			if (this.nodes.length > 0) {
				const message = magicai_localize['you_have_unsaved_changes__are_you_sure_you_want_to_leave_'];

				event.preventDefault();

				event.returnValue = message;

				return message;
			}
		},

		addBeforeUnloadListener() {
			this.beforeUnloadListened = true;
			window.addEventListener('beforeunload', this.beforeUnloadHandler);
		},

		removeBeforeUnloadListener() {
			this.beforeUnloadListened = false;
			window.removeEventListener('beforeunload', this.beforeUnloadHandler);
		},

		handleExport() {
			if (!this.stage || !this.nodes.length) return;

			try {
				const exportData = this.getExportData();

				const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(exportData));
				const downloadAnchorNode = document.createElement('a');
				downloadAnchorNode.setAttribute('href', dataStr);
				downloadAnchorNode.setAttribute('download', `canvas-export-${new Date().getTime()}.json`);
				document.body.appendChild(downloadAnchorNode);
				downloadAnchorNode.click();
				downloadAnchorNode.remove();

				toastr.success('Canvas exported successfully');
			} catch (error) {
				console.error('Error exporting canvas:', error);
				toastr.error('Failed to export canvas');
			}
		},

		getExportData() {
			const nodesToExport = this.nodes.map(node => node.toJSON());

			return {
				name: this.currentDocName || 'Untitled Document',
				stage: {
					width: this.stage.width(),
					height: this.stage.height(),
				},
				nodes: nodesToExport
			};
		},

		handleImport() {
			const fileInput = document.createElement('input');
			fileInput.type = 'file';
			fileInput.accept = '.json';
			fileInput.style.display = 'none';

			fileInput.addEventListener('change', event => {
				const file = event.target.files[0];
				if (!file) return;

				const reader = new FileReader();
				reader.onload = e => {
					try {
						const data = JSON.parse(e.target.result);

						this.importData({ data });

						this.currentView !== 'editor' && this.switchView('editor');

						toastr.success('Canvas imported successfully');
					} catch (error) {
						console.error('Error importing canvas:', error);
						toastr.error('Failed to import canvas');
					}
				};
				reader.readAsText(file);

				fileInput.remove();
			});

			document.body.appendChild(fileInput);
			fileInput.click();
		},

		importData({ data, docId = null, switchToEditor = true }) {
			if (!data.nodes || !Array.isArray(data.nodes)) {
				throw new Error('Invalid file format');
			}

			if (
				!this.nodes.length ||
				(
					this.nodes.length &&
					confirm(magicai_localize['all_your_current_edits_will_be_lost__are_you_sure_'])
				)
			) {
				this.destroyNodes({ nodes: 'all' });

				if ( data.name ) {
					this.currentDocName = data.name;
				}

				if (data.stage) {
					this.handleStageResize({
						width: data.stage.width,
						height: data.stage.height
					});
				}

				data.nodes.forEach(nodeData => {
					try {
						const tempNode = Konva.Node.create(nodeData);
						const className = tempNode.getClassName();
						const attrs = tempNode.getAttrs();

						this.addNodeToStage({
							type: className,
							attrs: attrs,
							addHistory: false,
							beforeAdd: node => {
								if ( attrs.fontFamily ) {
									this.loadGoogleFontFull(attrs.fontFamily);
								}

								if (className === 'Group') {
									const children = tempNode.getChildren();
									children.forEach(child => {
										child.draggable(false);
										node.add(child.clone());
									});
								}
							}
						});

						tempNode.destroy();
					} catch (nodeError) {
						console.error('Error creating node:', nodeError);
					}
				});

				this.resetHistory();

				this.currentDocId = docId;

				switchToEditor && this.currentView !== 'editor' && this.switchView('editor');
			}
		},

		async loadTemplatesList(url) {
			if ( url == null ) return;

			this.loadingTemplates = true;
			this.loadingTemplatesFailed = false;

			const res = await fetch(url);
			const data = await res.json();

			this.loadingTemplates = false;

			if ( !(data instanceof Array) || !data.length ) {
				this.loadingTemplatesFailed = true;
				toastr.error(magicai_localize['could_not_fetch_templates_list_please_try_again'] || 'Could not fetch templates list. Please try again.');
				return;
			}

			this.templatesList = data;
		},

		loadTemplate(templateId) {
			if ( templateId == null ) return;

			const templateData = this.templatesList.find(item => item.id == templateId);

			if ( !templateData ) {
				return toastr.error(magicai_localize['could_not_fetch_template_data_please_try_again'] || 'Could not fetch template data. Please try again.');
			}

			this.importData({ data: templateData.data });
		},

		async saveDocument() {
			const data = this.getExportData();
			const formData = new FormData();

			const container = document.createElement('div');
			container.style.position = 'absolute';
			container.style.zIndex = '-1';
			container.style.opacity = '0';
			container.style.width = `${this.stage.width()}px`;
			container.style.height = `${this.stage.height()}px`;
			document.body.appendChild(container);
			const tempStage = new Konva.Stage({
				container: container,
				width: this.stage.width(),
				height: this.stage.height(),
				background: 'white'
			});
			const clonedLayer = this.layer.clone();

			clonedLayer.find('#lqd-cs-selection-transformer, #lqd-cs-selection-rectangle').forEach(node => {
				node.remove();
			});

			tempStage.add(clonedLayer);

			const blob = await tempStage.toBlob({
				mimeType: 'image/png',
				quality: 0.7,
				pixelRatio: Math.min(2, Math.max(1, 720 / Math.min(720, tempStage.width())))
			});

			document.body.removeChild(container);
			tempStage.destroy();

			formData.append('payload', JSON.stringify(data));
			formData.append('preview', new File([ blob ], 'preview.png', { type: 'image/png' }));

			if ( this.currentDocId != null ) {
				formData.append( 'id', this.currentDocId );
			}

			if ( this.currentDocName ) {
				formData.append( 'name', this.currentDocName );
			}

			try {
				const response = await fetch('/dashboard/user/creative-suite/document', {
					method: 'POST',
					body: formData,
					headers: {
						'Accept': 'application/json'
					}
				});

				if (!response.ok) {
					throw new Error(`HTTP error! status: ${response.status}`);
				}

				const responseData = await response.json();

				if (responseData.status === 'error') {
					toastr.error(responseData.message || magicai_localize['error_saving_document']);
					return;
				}

				if (responseData.data.id) {
					const docEls = document.querySelectorAll(`.lqd-cs-doc-item[data-id="${responseData.data.id}"]`);
					this.currentDocId = responseData.data.id;

					if ( !docEls.length ) {
						this.newItems.push(responseData.data);
					} else {
						docEls.forEach(el => el.querySelector('img').src = responseData.data.preview_url);
					}
				}

				this.removeBeforeUnloadListener();

				toastr.success(magicai_localize['saved_succesfully']);
			} catch (error) {
				toastr.error('Upload failed:', error);
			}
		},

		async duplicateDocument(docId) {
			if ( !docId ) return;

			const formData = new FormData();

			formData.append('id', docId);

			try {
				const response = await fetch('/dashboard/user/creative-suite/document/duplicate', {
					method: 'POST',
					body: formData,
					headers: {
						'Accept': 'application/json'
					}
				});

				if (!response.ok) {
					throw new Error(`HTTP error! status: ${response.status}`);
				}

				const responseData = await response.json();

				if (responseData.status === 'error') {
					toastr.error(responseData.message || magicai_localize['error_saving_document']);
					return;
				}

				if (responseData.data) {
					if ( !document.querySelector(`.lqd-cs-doc-item[data-id="${responseData.data.id}"]`) ) {
						this.newItems.push(responseData.data);
					}
					toastr.success(magicai_localize['duplicated_the_document_successfully']);
				}
			} catch (error) {
				toastr.error('Upload failed:', error);
			}
		},

		async deleteDocument(docId) {
			if ( !docId ) return;

			const formData = new FormData();

			formData.append('id', docId);


			try {
				const response = await fetch('/dashboard/user/creative-suite/document/delete', {
					method: 'POST',
					body: formData,
					headers: {
						'Accept': 'application/json'
					}
				});

				if (!response.ok) {
					throw new Error(`HTTP error! status: ${response.status}`);
				}

				const responseData = await response.json();

				if (responseData.status === 'error') {
					toastr.error(responseData.message || magicai_localize['error_saving_document']);
					return;
				}

				if (responseData.status === 'success') {
					document.querySelectorAll(`.lqd-cs-doc-item[data-id="${docId}"]`)?.forEach(el => el.remove());
					toastr.success(magicai_localize['document_deleted_successfully']);
				}
			} catch (error) {
				toastr.error('Upload failed:', error);
			}
		},

		async loadDocument(docId) {
			if ( !docId ) return;

			try {
				const response = await fetch(`/dashboard/user/creative-suite/document/${docId}`);

				if (!response.ok) {
					throw new Error(`HTTP error! status: ${response.status}`);
				}

				const data = await response.json();

				if ( !data.data.payload ) return;

				this.importData({ data: JSON.parse(data.data.payload), docId: data.data.id });

				this.currentDocName = data.data.name;

				this.$nextTick(() => {
					this.positionTooltip();
				});
			} catch (error) {
				console.error('Upload failed:', error);
			}
		},

		async updateDocumentName(event) {
			const formData = new FormData(event.target);

			if ( !formData.get('id') ) return;

			this.updatingName = true;

			try {
				const response = await fetch('/dashboard/user/creative-suite/document/name', {
					method: 'POST',
					body: formData,
					headers: {
						'Accept': 'application/json'
					}
				});

				if (!response.ok) {
					throw new Error(`HTTP error! status: ${response.status}`);
				}

				const data = await response.json();

				if (data.status === 'error') {
					toastr.error(responseData.message || magicai_localize['error_saving_document']);
					return;
				}

				const els = document.querySelectorAll(`.lqd-cs-doc-item[data-id="${data.id}"]`);

				els.forEach(el => {
					const nameEl = el.querySelector('h5');

					if ( nameEl ) {
						nameEl.innerText = data.name;
					}
				});

				toastr.success(magicai_localize['name_updated_successfully']);
			} catch (error) {
				toastr.error('Updating name failed:', error);
			} finally {
				this.updatingName = false;
			}

		},

		handleWindowResize() {
			window.addEventListener('resize', this.onWindowResize);
		},

		onWindowResize() {
			this.setContainerDimensions();
		},

		setActiveTooltipDropdown(dropdown) {
			if (!dropdown || this.activeTooltipDropdown === dropdown) {
				return this.activeTooltipDropdown = null;
			}

			this.activeTooltipDropdown = dropdown;
		},

		elementIsFocusable(element, excludes = []) {
			if ( !element || !(element instanceof HTMLElement) ) return;

			const elementsArray = [
				'INPUT',
				'BUTTON',
				'TEXTAREA',
				'SELECT',
				'A',
			];

			return difference(elementsArray, excludes).includes(element.tagName);
		},

		docActiveElementIsWritable() {
			return (
				document.activeElement &&
				[ 'TEXTAREA', 'INPUT' ].includes(document.activeElement.tagName)
			);
		},

		async aiTextAction({ prompt = '' }) {
			if ( !prompt.trim() ) {
				return;
			}

			const textNodes = this.selectedNodes.filter(node => node.getClassName() === 'Text');

			if ( !textNodes.length ) return;

			textNodes.forEach(async node => {
				const lockKey = `request-${node.id()}`;
				const formData = new FormData();

				formData.append( 'lock_key', lockKey );
				formData.append( 'prompt', prompt );
				formData.append( 'content', node.text() );

				this.aiTasksQueue.push(lockKey);

				node.setAttr('aiTaskInProgress', true);

				const res = await fetch('/dashboard/user/openai/update-writing', {
					method: 'POST',
					body: formData,
					headers: {
						'Accept': 'application/json'
					}
				});

				this.aiTasksQueue = this.aiTasksQueue.filter(q => q === lockKey);

				node.setAttr('aiTaskInProgress', false);

				const data = await res.json();

				if (data.type === 'error') {
					toastr.error(data.message);
					return;
				}

				if ( !data.result ) return;

				node.text(data.result);
			});
		},

		async aiImageAction(event) {
			const imageNodes = this.selectedNodes.filter(node => node.getClassName() === 'Image' || node.fillPatternImage());

			if ( !imageNodes.length ) return;

			imageNodes.forEach(async node => {
				const lockKey = `request-${node.id()}`;
				const formData = new FormData(event.target);
				const nodeWidth = Math.min(720, node.width());
				const fillImageWidth = Math.min(720, (node.getAttr('fillImageWidth') ?? 720));

				const nodeblob = await node.toBlob({
					pixelRatio: Math.min(2, Math.max(1, fillImageWidth / nodeWidth))
				});
				const imageFile = new File([ nodeblob ], 'image', { type: nodeblob.type });

				formData.append( 'lock_key', lockKey );
				formData.append( 'uploaded_image', imageFile );

				this.aiTasksQueue.push(lockKey);

				node.setAttr('aiTaskInProgress', true);

				const res = await fetch(event.target.action, {
					method: 'POST',
					body: formData,
					headers: {
						'Accept': 'application/json'
					}
				});

				this.aiTasksQueue = this.aiTasksQueue.filter(q => q === lockKey);

				node.setAttr('aiTaskInProgress', false);

				if (!res.ok) {
					return res.json().then(errorData => {
						toastr.error(errorData.message || 'An unknown error occurred');
					});
				}

				const data = await res.json();

				if (data.type === 'error') {
					toastr.error(data.message);
					return;
				}

				if (data?.data?.status === 'CREATED' || data?.data?.status === 'IN_PROGRESS') {
					this.getAIImageStatus(data.data, node, lockKey);
				} else {
					this.onAIImageProcessDone(data.data, node, lockKey);
				}

			});
		},

		async getAIImageStatus(data = {}, node, lockKey) {
			const res = await fetch('/dashboard/user/advanced-image/editor/' + data.id + '/status', {
				method: 'GET',
				headers: {
					'Accept': 'application/json'
				}
			});

			if (!res.ok) {
				return res.json().then(errorData => {
					toastr.error(errorData.message || 'An unknown error occurred');
				});
			}

			const resData = await res.json();

			if (resData.status === 'error') {
				return toastr.error(resData.message);
			}

			if (resData.data.status === 'COMPLETED') {
				this.onAIImageProcessDone(resData.data, node, lockKey);
			} else {
				setTimeout(() => {
					this.getAIImageStatus(resData.data, node, lockKey);
				}, 1000);
			}
		},

		onAIImageProcessDone(data, node, lockKey) {
			console.log(data);
		},

		isMobile() {
			return window.innerWidth < 768;
		},

		togglePromptLibraryShow() {
			this.promptLibraryShow = !this.promptLibraryShow;
		},
		changePromptFilter(filter) {
			filter !== this.promptFilter && (this.promptFilter = filter);
		},
		setSearchPromptStr(str) {
			this.searchPromptStr = str.trim().toLowerCase();
		},
		setPrompt(prompt) {
			this.prompt = prompt;
		},
		focusOnPrompt() {
			this.$nextTick(() => {
				this.$refs.promptInput?.focus();
				this.dropdownOpen = true;
			});
		},
	});
};
