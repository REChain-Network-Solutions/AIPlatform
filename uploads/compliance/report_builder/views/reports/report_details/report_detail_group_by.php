<table id="dtBasicExample"  class="table dt-table border ">
	<thead>
		<th class="sorting_disabled hide"><?php echo _l('ID'); ?></th>


		<?php $header_key = []; ?>
		<?php foreach ($get_selected_column as $report_key => $report_header) { ?>
			<?php 
				$header_key[] = db_prefix().$report_header['table_name'].'.'.$report_header['field_name'];
			 ?>
			<th><?php echo new_html_entity_decode($report_header['label_name']); ?></th>
		<?php } ?>
	</thead>

	<tbody>
		<?php $index = 1;?>

		<?php foreach ($report_result as $key => $report_value) { ?>

			<?php 


			$report_children_data = rb_report_get_children_data($id, $search, $group_by_columns, $report_value);

			?>

			<!-- children data -->
			<?php if(count($report_children_data) > 0){ ?>
				<?php foreach ($report_children_data as $key => $children_value) { ?>
					<tr>
						<td class="hide"><b><?php echo new_html_entity_decode($index); ?></b></td>

						<?php foreach ($children_value as $children_key => $value) { ?>

							<td style="<?php echo rb_cell_formatting_color($children_key, $cell_formattings, $value) ?>"><?php echo new_html_entity_decode(rb_report_detail_format_value($value, $children_key)); ?></td>

						<?php } ?>
					</tr>
					<?php $index++;?>


				<?php } ?>
			<?php } ?>

			<!-- parent data -->
				<?php if($allow_subtotal){ ?>
			<tr class="font-color-red" >
				<td class="hide"><b><?php echo new_html_entity_decode($index); ?></b></td>
				<?php foreach ($header_key as $header_index => $header_value) { ?>
					<?php 
						$sub_total_value ='';
						if($header_index == 0){
							$sub_total_value .= _l('rb_subtotal');
						}


						if($header_index != 0){
							if(isset($report_value[$header_value])){
								$sub_total_value .= ' '.rb_report_detail_format_value($report_value[$header_value], $header_value);

							}elseif(isset($report_value['sum_'.$header_value])){
								$sub_total_value .= ' '.rb_report_detail_format_value($report_value['sum_'.$header_value], $header_value);

							}elseif(isset($report_value['count_'.$header_value])){
								$sub_total_value .= ' '.rb_report_detail_format_value($report_value['count_'.$header_value], $header_value);

							}elseif(isset($report_value['average_'.$header_value])){
								$sub_total_value .= ' '.rb_report_detail_format_value($report_value['average_'.$header_value], $header_value);

							}elseif(isset($report_value['min_'.$header_value])){

								$sub_total_value .= ' '.rb_report_detail_format_value($report_value['min_'.$header_value], $header_value);
							}elseif(isset($report_value['max_'.$header_value])){

								$sub_total_value .= ' '.rb_report_detail_format_value($report_value['max_'.$header_value], $header_value);

							}

							
						}

					 ?>
					<td class="background-aliceblue" ><?php echo new_html_entity_decode($sub_total_value); ?></td>

				<?php } ?>

				<?php $index++;?>

			</tr>
				<?php } ?>


		<?php } ?>



	</tbody>
</table> 

