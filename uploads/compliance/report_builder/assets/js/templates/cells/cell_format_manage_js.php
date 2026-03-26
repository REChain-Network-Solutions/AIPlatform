
<script>

	(function($) {
		"use strict";

		var InvoiceServerParams={
			"report_template_id": "[name='report_template_id']",
		};
		var cell_format_table = $('.table-cell_format_table');
		initDataTable(cell_format_table, admin_url+'report_builder/cell_format_table',[0],[0], InvoiceServerParams, [0 ,'desc']);
		
	})(jQuery); 

	function add_cell_formatting(cell_formatting_id, template_id, add_new) {
	"use strict";

	  $("#modal_wrapper").load("<?php echo admin_url('report_builder/report_builder/cell_formatting_modal'); ?>", {
	       slug: add_new,
	       template_id: template_id,
	       cell_formatting_id: cell_formatting_id,
	  }, function() {
	  	$("body").find('#cellFormattingModal').modal({ show: true, backdrop: 'static' });

	  });

	  init_selectpicker();
	  init_color_pickers();
	  $(".selectpicker").selectpicker('refresh');
	}

	$('.data_source_next').on('click', function() {
		"use strict";

		var template_id = $("input[name='report_template_id']").val();
		var type = 'cell_formatting';
		$.get(admin_url + 'report_builder/get_next_step_report/' + template_id+'/'+type, function (response) {

			if(response.status == true || response.status == 'true'){
				window.location.assign(response.next_link);
			}else{
				alert_float('danger', '<?php echo _l('please_add_relationship_between_table'); ?>');
			}

		}, 'json');

	});

	function field_name_change() {
		"use strict";

		//get filter type from field name
		var data = {};
		data.table_name = $("select[name='table_name']").val();
		data.field_name = $("select[name='field_name']").val();
		data.cell_formatting = true;
		data.filter_type_selected = '';


		$.get(admin_url + 'report_builder/get_filter_type', data, function (response) {

			$("select[name='filter_type']").html('');
			$("select[name='filter_type']").append(response.filter_type);

			init_datepicker();
			init_selectpicker();
			$(".selectpicker").selectpicker('refresh');
		}, 'json');
	}

</script>