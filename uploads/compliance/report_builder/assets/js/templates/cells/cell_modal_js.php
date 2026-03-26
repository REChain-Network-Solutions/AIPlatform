<script>
	(function($) {
		"use strict";
		
		init_color_pickers();
		init_selectpicker();
		init_datepicker();
		
		$(".selectpicker").selectpicker('refresh');

		appValidateForm($("body").find('#add_cell_formatting'), {
			'field_name': 'required',
			'filter_type': 'required',
		},submit_filter);

		$('select[name="table_name"]').on('change', function() {
			"use strict";

			var table_name =$(this).val();

			$.get(admin_url + 'report_builder/get_list_fields/' + table_name, function (response) {

				$("select[name='field_name']").html('');
				$("select[name='field_name']").append(response.list_field_options);

				$("select[name='filter_type']").val('').change();
				$("input[name='filter_value_1']").val('');
				$("input[name='filter_value_2']").val('');

				init_selectpicker();
				$(".selectpicker").selectpicker('refresh');
			}, 'json');

		});

	})(jQuery); 
	
	var table_cell_formatting = $('table.table-cell_format_table');

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

			$('#cellFormattingModal').modal('hide');

			table_cell_formatting.DataTable().ajax.reload(null, false);
		});

	}

	$('select[name="field_name"]').on('change', function() {
		"use strict";

		field_name_change();
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
		data.cell_formatting = true;


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