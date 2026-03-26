
<script>

	(function($) {
		"use strict";  

		var InvoiceServerParams={
			"report_template_id": "[name='report_template_id']",
		};
		var filter_table = $('.table-filter_table');
		initDataTable(filter_table, admin_url+'report_builder/filter_table',[0],[0], InvoiceServerParams, [0 ,'desc']);

		$('#date_add').on('change', function() {
			filter_table.DataTable().ajax.reload().columns.adjust().responsive.recalc();
		});

		var hidden_columns = [0];
		$('.table-filter_table').DataTable().columns(hidden_columns).visible(false, false);
		
	})(jQuery);

	/**
	 * add filter
	 * @param {[type]} template_id 
	 * @param {[type]} add_new     
	 */
	function add_filter(template_id, slug, datasource_filter_id) {
	"use strict";

	  $("#modal_wrapper").load("<?php echo admin_url('report_builder/report_builder/filter_modal'); ?>", {
	       datasource_filter_id: datasource_filter_id,
	       slug: slug,
	       template_id: template_id,
	  }, function() {
	  	$("body").find('#filterModal').modal({ show: true, backdrop: 'static' });
	  });

	  init_selectpicker();
	  $(".selectpicker").selectpicker('refresh');
	}

</script>