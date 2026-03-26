<script>
var lastAddedDeliveryItemKey = null;

	$(".selectpicker").selectpicker('refresh');


	$('select[name="right_table"]').on('change', function() {
		"use strict";

		var left_table =$(this).val();

		$.get(admin_url + 'report_builder/get_table_related_data/' + left_table, function (response) {

			$("select[name='right_field_1']").html('');
			$("select[name='right_field_1']").append(response.left_field_option_1);

			$("select[name='right_field_2']").html('');
			$("select[name='right_field_2']").append(response.left_field_option_2);
			
			init_selectpicker();
			$(".selectpicker").selectpicker('refresh');
		}, 'json');
		
	});

	$('select[name="left_field_1"]').on('change', function() {
		"use strict";

		generate_query_string();
	});

	$('select[name="right_field_1"]').on('change', function() {
		"use strict";

		generate_query_string();
	});

	$('select[name="left_field_2"]').on('change', function() {
		"use strict";

		generate_query_string();
	});

	$('select[name="right_field_2"]').on('change', function() {
		"use strict";

		generate_query_string();
	});

	function generate_query_string() {
		"use strict";

		var query_tring='';

		var left_table = $("select[name='left_table']").val();
		var right_table = $("select[name='right_table']").val();
		var left_field_1 = $("select[name='left_field_1']").val();
		var right_field_1 = $("select[name='right_field_1']").val();
		var left_field_2 = $("select[name='left_field_2']").val();
		var right_field_2 = $("select[name='right_field_2']").val();

		if(left_table.length != 0 && right_table.length != 0 && left_field_1.length != 0 && right_field_1.length != 0){
			if(right_field_2.length != 0 && left_field_2.length != 0){
				query_tring += left_table+'.'+left_field_1+' = '+right_table+'.'+right_field_1+' AND '+left_table+'.'+left_field_2+' = '+right_table+'.'+right_field_2;
			}else{
				query_tring += left_table+'.'+left_field_1+' = '+right_table+'.'+right_field_1;
			}
		}

		$("textarea[name='query_string']").val(query_tring);
	}

	var table_relationship = $('table.table-relationship_table');

	

	function get_item_data(table_name, column){
		"use strict";

		$("body").append('<div class="dt-loader"></div>');
		get_column_data(table_name).done(function(response){

			response = JSON.parse(response);

			$('select[id="'+column+'"]').html('');
			$('select[id="'+column+'"]').val(response.column_name).change();


		console.log('table_value', $('select[id="'+table_name+'"]').val());

			$("body").find('.dt-loader').remove();
			return true;
		}, 'json');
	}


	function get_column_data(table_name){
		"use strict";

		jQuery.ajaxSetup({
			async: false
		});
		console.log('table_value', $('select[id="'+table_name+'"]'));
		
		var d = $.post(site_url + 'admin/report_builder/get_column_data', {
			table_name: $('select[id="'+table_name+'"]').val(),
		});
		jQuery.ajaxSetup({
			async: true
		});
		return d;
	}



	function table_onchange(table_name, type_of_join, query_string, main_table=''){
		"use strict";


		var array_type_of_join = [];
		var array_table_name = [];
		var array_type_of_join_data = [];
		var rows = $('.relationship_data').find('.relationship_row');
		$.each(rows, function() {

			var type_of_join_value = $($(this).find('div.type_of_join').eq(0)).find('select');
			var table_name_value = $($(this).find('div.table_name').eq(0)).find('select');
			var type_of_join_data_value = $($(this).find('div.type_of_join_data').eq(0)).find('select');

			array_type_of_join.push(type_of_join_value.val());
			array_table_name.push(table_name_value.val());
			array_type_of_join_data.push(type_of_join_data_value.val());

		});

		var table_name_value = $('select[id="'+table_name+'"]').val();

		$("body").append('<div class="dt-loader"></div>');
		get_type_of_join_data(table_name, type_of_join, query_string, array_type_of_join, array_table_name, array_type_of_join_data, main_table).done(function(response){

			response = JSON.parse(response);

			$('select[id="'+query_string+'"]').html('');
			$('select[id="'+query_string+'"]').html(response.type_of_join_data);

			if(table_name == 'newitems[left_table]'){
				$('select[id="left_table"]').html('');
				$('select[id="left_table"]').html(response.related_table_data_html);
			}

			if(main_table == 1 && table_name_value == ''){
				$('select[id="'+table_name+'"]').html('');
				$('select[id="'+table_name+'"]').html(response.related_table_data_html);
			}

			if(main_table == 1 && table_name_value != '' && rows.length == 2){
				$('select[id="left_table"]').html('');
				$('select[id="left_table"]').html(response.related_table_data_html);
			}

			$("body").find('.dt-loader').remove();
			return true;
		}, 'json');

	  $(".selectpicker").selectpicker('refresh');

	}

	function get_type_of_join_data(table_name, type_of_join, query_string, array_type_of_join, array_table_name, array_type_of_join_data, main_table){
		"use strict";

		jQuery.ajaxSetup({
			async: false
		});
		
		var d = $.post(site_url + 'admin/report_builder/get_type_of_join_data', {
			table_names: $('select[id="'+table_name+'"]').val(),
			type_of_joins: type_of_join,
			query_strings: query_string,
			array_type_of_joins: array_type_of_join,
			array_table_names: array_table_name,
			array_type_of_join_datas: array_type_of_join_data,
			main_table: main_table,

		});

		jQuery.ajaxSetup({
			async: true
		});
		return d;
	}

	function get_item_preview_values() {
		"use strict";
		var response = {};

		response.type_of_join = $('.main select[name="join_type"]').val();
		response.table_name = $('.main select[name="left_table"]').val();
		response.type_of_join_data = $('.main select[name="query_string"]').val();

		return response;
	}

	function clear_close_shift_item_preview_values(parent) {
		"use strict";

		var previewArea = $(parent + ' .main');
		previewArea.find('input').val('');
		previewArea.find('select').val('').change();
	}

	function reorder_close_shift_items(parent) {
		"use strict";

		var rows = $(parent + ' div.item');
		var i = 1;
		$.each(rows, function () {
			$(this).find('input.order').val(i);
			i++;
		});
	}

	function add_relationship_item_to_table(data, query_string) {
		"use strict";

		data = typeof (data) == 'undefined' || data == 'undefined' ? get_item_preview_values() : data;
		if (data.type_of_join == "" || data.table_name == "" || data.type_of_join_data == "" ) {
			return;
		}

		//get list related table start
		var related_table_data_html='';
		var array_type_of_join = [];
		var array_table_name = [];
		var array_type_of_join_data = [];
		var this_array_type_of_join_data = [];

		var rows = $('.relationship_data').find('.relationship_row');
		$.each(rows, function() {

			var type_of_join_value = $($(this).find('div.type_of_join').eq(0)).find('select');
			var table_name_value = $($(this).find('div.table_name').eq(0)).find('select');
			var type_of_join_data_value = $($(this).find('div.type_of_join_data').eq(0)).find('select');

			array_type_of_join.push(type_of_join_value.val());
			array_table_name.push(table_name_value.val());
			array_type_of_join_data.push(type_of_join_data_value.val());

		});


		$("body").append('<div class="dt-loader"></div>');
		get_type_of_join_data('left_table', data.type_of_join, data.type_of_join_data, array_type_of_join, array_table_name, array_type_of_join_data).done(function(response){
			response = JSON.parse(response);

			related_table_data_html = response.related_table_data_html;
			this_array_type_of_join_data = response.array_type_of_join_data;
			

		}, 'json');

		//get list related table end


		var table_row = '';
		var item_key = lastAddedDeliveryItemKey ? lastAddedDeliveryItemKey += 1 : $("body").find('.relationship_data div').length + 1;
		lastAddedDeliveryItemKey = item_key;
		$("body").append('<div class="dt-loader"></div>');
		get_relationship_row_template('newitems[' + item_key + ']',data.type_of_join,data.table_name,data.type_of_join_data,item_key, array_table_name, this_array_type_of_join_data).done(function(output){
			table_row += output;

			$('.relationship_data div.main').before(table_row);

			reorder_close_shift_items('.relationship_data');
			clear_close_shift_item_preview_values('.relationship_data');
			$('body').find('#items-warning').remove();
			$("body").find('.dt-loader').remove();

			$('select[id="left_table"]').html('');
			$('select[id="left_table"]').html(related_table_data_html);
			$(".selectpicker").selectpicker('refresh');
			
			return true;
		});
		return false;

	}

	function get_relationship_row_template(name, type_of_join,table_name, type_of_join_data,item_id, array_table_name, this_array_type_of_join_data) {
		"use strict";

		jQuery.ajaxSetup({
			async: false
		});
		var d = $.post(site_url + 'admin/report_builder/get_relationship_row_template', {
			name: name,
			type_of_join: type_of_join,
			table_name: table_name,
			type_of_join_data: type_of_join_data,
			item_id: 'undefined',
			array_table_name: array_table_name,
			this_array_type_of_join_data: this_array_type_of_join_data,
		});
		jQuery.ajaxSetup({
			async: true
		});
		return d;
	}

	function delete_relationship_item(row, itemid, parent) {
		"use strict";

		$(row).parents('div.relationship_row').remove();
		
		if (itemid) {
			$('#removed-items').append(hidden_input('removed_items[]', itemid));
		}
	}

</script>


