<script>
	(function($) {
		"use strict";  
		init_datepicker();
		init_selectpicker();
		$(".selectpicker").selectpicker('refresh');

		appValidateForm($("body").find('#add_filter'), {

		},submit_filter);

	})(jQuery); 

	$('select[name="table_name"]').on('change', function() {
		"use strict";

		var table_name =$(this).val();

		if(table_name != ''){
			$.get(admin_url + 'report_builder/get_list_fields/' + table_name, function (response) {

				$("select[name='field_name']").html('');
				$("select[name='field_name']").append(response.list_field_options);

			init_selectpicker();
			$(".selectpicker").selectpicker('refresh');
		}, 'json');
		}

	});



	$('select[name="field_name"]').on('change', function() {
		"use strict";

		//get filter type from field name
		var data = {};
		data.table_name = $("select[name='table_name']").val();
		data.field_name = $(this).val();

		$.get(admin_url + 'report_builder/get_filter_type', data, function (response) {

			$("select[name='filter_type']").html('');
			$("select[name='filter_type']").append(response.filter_type);

			init_selectpicker();
			$(".selectpicker").selectpicker('refresh');
		}, 'json');

	});


	$('input[name="filter_value_1"]').on('change', function() {
		"use strict";

		generate_filter_string();
	});

	$('input[name="filter_value_2"]').on('change', function() {
		"use strict";

		generate_filter_string();
	});

	function generate_filter_string() {
		"use strict";

		var filter_tring='';

		var table_name = $("select[name='table_name']").val();
		var field_name = $("select[name='field_name']").val();
		var filter_type = $("select[name='filter_type']").val();
		var filter_value_1 = $("input[name='filter_value_1']").val();
		var filter_value_2 = $("input[name='filter_value_2']").val();

		if(table_name.length != 0 && field_name.length != 0 && filter_type != null && filter_type.length != 0 && filter_value_1.length != 0){

			switch(filter_type) {
			  case 'equal':
			    	filter_tring += table_name+'.'+field_name+' = '+filter_value_1;
			    break;

			    case 'greater_than':
			    // code block
			    	filter_tring += table_name+'.'+field_name+' > '+filter_value_1;
			    break;

			    case 'less_than':
			    // code block
			    	filter_tring += table_name+'.'+field_name+' < '+filter_value_1;
			    break;

			    case 'greater_than_or_equal':
			    // code block
			    	filter_tring += table_name+'.'+field_name+' >= '+filter_value_1;
			    break;

			    case 'less_than_or_equal':
			    // code block
			    	filter_tring += table_name+'.'+field_name+' <= '+filter_value_1;
			    break;

			    case 'between':
			    // code block
			    	filter_tring += table_name+'.'+field_name+' BETWEEN '+filter_value_1+' AND '+filter_value_2 ;
			    break;

			    case 'like':
			    // code block
			    	filter_tring += table_name+'.'+field_name+' LIKE '+filter_value_1;
			    break;

			    case 'NOT_like':
			    // code block
			    	filter_tring += table_name+'.'+field_name+' NOT LIKE '+filter_value_1;
			    break;

			    case 'not_equal':
			    // code block
			    	filter_tring += table_name+'.'+field_name+' != '+filter_value_1;
			    break;

			    case 'begin_with':
			    // code block
			    	filter_tring += table_name+'.'+field_name+' LIKE '+filter_value_1+'%';
			    break;

			    case 'end_with':
			    // code block
			    	filter_tring += table_name+'.'+field_name+' LIKE '+'%'+filter_value_1;
			    break;

			    case 'in':
			    // code block
			    	filter_tring += table_name+'.'+field_name+' IN ('+filter_value_1+')';
			    break;

			    case 'not_in':
			    // code block
			    	filter_tring += table_name+'.'+field_name+' NOT IN ('+filter_value_1+')';
			    break;
			}
		}

		$("textarea[name='query_filter']").val(filter_tring);
	}

	var table_filter = $('table.table-filter_table');

	function submit_filter(form) {
		"use strict";

		var data={};
		data.formdata = $( form ).serializeArray();

		$.post(form.action, data).done(function(response) {
			var response = JSON.parse(response);
			if(response.status == 'true'){
				alert_float('success', response.message);
			}else{
				alert_float('danger', response.message);
			}

			$('#filterModal').modal('hide');

			table_filter.DataTable().ajax.reload(null, false);
		});

	}

	$('input[name="ask_user"]').on('click', function() {
		"use strict";

		var ask_user = $('input[id="ask_user"]').is(":checked");
		if(ask_user == true){

			$(".filter_input").html('');

		}else{

			get_filter_value();
			$(".filter_input").removeClass('hide');

		}

	});

	$('select[name="filter_type"]').on('change', function() {
		"use strict";

		get_filter_value();

	});


	function get_filter_value() {
		"use strict";
		
		//get filter type from field name
		var data = {};
		data.table_name = $("select[name='table_name']").val();
		data.field_name = $("select[name='field_name']").val();
		data.filter_type = $("select[name='filter_type']").val();

		$.get(admin_url + 'report_builder/get_datasource_filter_value', data, function (response) {

			$(".filter_input").html('');
			$(".filter_input").append(response.filter_value);

			if($('input[id="ask_user"]').is(":checked")){
				$('input[id="ask_user"]').prop( "checked", false );
			}

			init_datepicker();
			init_selectpicker();
			$(".selectpicker").selectpicker('refresh');
		}, 'json');
	}

</script>