
<script>

	(function($) {
		"use strict"; 

		var InvoiceServerParams={
			"category_filter"    : "select[name='category_filter[]']",
			"staff_filter"    : "select[name='staff_filter[]']",
		};
		var report_table = $('.table-report_table');
		initDataTable(report_table, admin_url+'report_builder/report_table',[0],[0], InvoiceServerParams, [0 ,'desc']);


		$('#category_filter').on('change', function() {
			report_table.DataTable().ajax.reload();
		});
		$('#staff_filter').on('change', function() {
			report_table.DataTable().ajax.reload();
		});
		

		var hidden_columns = [0];
		$('.table-report_table').DataTable().columns(hidden_columns).visible(false, false);
		
	})(jQuery); 

	/**
	 * report filter modal
	 * @param  {[type]} template_id 
	 * @return {[type]}             
	 */
	function report_filter_modal(template_id) {
	"use strict";


	  $("#modal_wrapper").load("<?php echo admin_url('report_builder/report_builder/report_filter_modal'); ?>", {
	       template_id: template_id,
	  }, function() {
	  	$("body").find('#filterModal').modal({ show: true, backdrop: 'static' });
	  });

	  init_selectpicker();
	  $(".selectpicker").selectpicker('refresh');
	}

	function sharing_modal(template_id, report) {
		"use strict";

		$("#modal_wrapper").load("<?php echo admin_url('report_builder/report_builder/sharing_modal'); ?>", {
			template_id: template_id,
			report: true,
		}, function() {
			$("body").find('#sharingModal').modal({ show: true, backdrop: 'static' });
		});

		init_selectpicker();
		$(".selectpicker").selectpicker('refresh');
	}

</script>