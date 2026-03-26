
<script>

	(function($) {
		"use strict";  

		var InvoiceServerParams={
			"category_filter"    : "select[name='category_filter[]']",
			"role_filter"    : "select[name='role_filter[]']",
			"department_filter"    : "select[name='department_filter[]']",
			"staff_filter"    : "select[name='staff_filter[]']",
		};
		var sharing_table = $('.table-sharing_table');
		initDataTable(sharing_table, admin_url+'report_builder/sharing_table',[0],[0], InvoiceServerParams, [0 ,'desc']);


		$('#category_filter').on('change', function() {
			sharing_table.DataTable().ajax.reload();
		});
		$('#role_filter').on('change', function() {
			sharing_table.DataTable().ajax.reload();
		});
		$('#department_filter').on('change', function() {
			sharing_table.DataTable().ajax.reload();
		});
		$('#staff_filter').on('change', function() {
			sharing_table.DataTable().ajax.reload();
		});
		

		var hidden_columns = [0];
		$('.table-sharing_table').DataTable().columns(hidden_columns).visible(false, false);
		
	})(jQuery); 

	/**
	 * report filter modal
	 * @param  {[type]} template_id 
	 * @return {[type]}             
	 */
	 function sharing_modal(template_id) {
	 	"use strict";

	 	$("#modal_wrapper").load("<?php echo admin_url('report_builder/report_builder/sharing_modal'); ?>", {
	 		template_id: template_id,
	 	}, function() {
	 		$("body").find('#sharingModal').modal({ show: true, backdrop: 'static' });
	 	});

	 	init_selectpicker();
	 	$(".selectpicker").selectpicker('refresh');
	 }


</script>