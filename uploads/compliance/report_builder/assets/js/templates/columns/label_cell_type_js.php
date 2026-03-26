<script>
	var purchase;

	<?php if(isset($columns)){?>

		(function($) {
			"use strict";  


			<?php if(isset($columns)){?>
				var dataObject_pu = <?php echo new_html_entity_decode($columns); ?>;
			<?php }else{ ?>
				var dataObject_pu = [];
			<?php } ?>

		//hansometable for purchase
		var row_global;
		var hotElement1 = document.getElementById('label_cell_type_hs');


		purchase = new Handsontable(hotElement1, {
			licenseKey: 'non-commercial-and-evaluation',

			contextMenu: true,
			manualRowMove: true,
			manualColumnMove: true,
			stretchH: 'all',
			autoWrapRow: true,
			rowHeights: 30,
			defaultRowHeight: 100,
			minRows: 10,
			maxRows: 40,
			width: '100%',

			rowHeaders: true,
			colHeaders: true,
			autoColumnSize: {
				samplingRatio: 23
			},

			filters: true,
			manualRowResize: true,
			manualColumnResize: true,
			allowInsertRow: true,
			allowRemoveRow: true,
			columnHeaderHeight: 40,

			colWidths:  [50, 120, 120, 120],
			rowHeights: 30,
			rowHeaderWidth: [44],
			minSpareRows: 1,
			hiddenColumns: {
				columns: [5,6],
				indicators: true
			},

			columns: [
			
			{
				type: 'numeric',
				data: 'order_display',

			},
			{
				type: 'text',
				data: 'table_name',
				readOnly: true

			},
			{
				type: 'text',
				data: 'field_name',
				readOnly: true

			},
			{
				type: 'text',
				data: 'label_name',
			},
			{

				type: 'text',
				data: 'field_type',
				renderer: customDropdownRenderer,
				editor: "chosen",
				chosenOptions: {
					data: <?php echo json_encode($rb_cell_type); ?>
				},

			},

			{
				type: 'text',
				data: 'templates_id',
				readOnly: true

			},
			{
				type: 'text',
				data: 'id',
				readOnly: true

			},
			

			],

			colHeaders: [
			'<?php echo _l('rb_order_display'); ?>',
			'<?php echo _l('table_name'); ?>',
			'<?php echo _l('field_name'); ?>',
			'<?php echo _l('rb_label_name'); ?>',
			'<?php echo _l('rb_field_type'); ?>',
			'<?php echo _l('templates_id'); ?>',
			'<?php echo _l('id'); ?>',
			],

			data: dataObject_pu,
		});


	})(jQuery);

	function customDropdownRenderer(instance, td, row, col, prop, value, cellProperties) {
		"use strict";
		var selectedId;
		var optionsList = cellProperties.chosenOptions.data;
		
		if(typeof optionsList === "undefined" || typeof optionsList.length === "undefined" || !optionsList.length) {
			Handsontable.cellTypes.text.renderer(instance, td, row, col, prop, value, cellProperties);
			return td;
		}

		var values = (value + "").split("|");
		value = [];
		for (var index = 0; index < optionsList.length; index++) {

			if (values.indexOf(optionsList[index].id + "") > -1) {
				selectedId = optionsList[index].id;
				value.push(optionsList[index].label);
			}
		}
		value = value.join(", ");

		Handsontable.cellTypes.text.renderer(instance, td, row, col, prop, value, cellProperties);
		return td;
	}

	var purchase_value = purchase;

	$('.add_label_cell_type').on('click', function() {
		'use strict';
		
		var valid_contract = $('#label_cell_type_hs').find('.htInvalid').html();

		if(valid_contract){
			alert_float('danger', "<?php echo _l('data_must_number') ; ?>");
		}else{

			$('input[name="label_cell_type_hs"]').val(JSON.stringify(purchase_value.getData()));   
			$('#add_label_cell_type').submit(); 

		}
	});

<?php } ?>



</script>