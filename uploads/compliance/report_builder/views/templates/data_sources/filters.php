<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>

<div>
	<div class="_buttons">
		<?php  if(is_admin() || has_permission('report_builder','','create')) { ?>
			<a href="#" onclick="add_filter(<?php echo new_html_entity_decode($report_template_id); ?>,'add', 0); return false;" class="btn btn-info mbot10"><?php echo _l('add_filter'); ?></a>
		<?php } ?>
	</div>
	
	<div class="clearfix"></div>
	<br>
	<div>
		<?php 
		$table_data = array(
			_l('id'),
			_l('query_filter'),
			_l('rb_group_condition'),
			_l('rb_ask_user'),
			_l('rb_options'),
			
		);

		render_datatable($table_data,'filter_table',
			array('customizable-table'),
			array(
				'id'=>'table-filter_table',
				'data-last-order-identifier'=>'filter_table',
				'data-default-order'=>get_table_last_order('filter_table'),
			)); 

			?>
	</div>
	<div id="modal_wrapper"></div>
</div>

</body>
</html>
