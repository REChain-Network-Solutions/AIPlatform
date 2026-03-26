<?php
defined('BASEPATH') or exit('No direct script access allowed');



/**
 * rb related table
 * @param  string $table 
 * @return [type]        
 */
function rb_related_table($table='')
{
	// primary key
	// foreign key
	// operator string
	
	//TODO
	$tables=[];

	//staff
	$tables[]=['name' => 'staff', 'label' => _l('tblstaff'), 'value' => [
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role', 'team_manage']],

		'roles' => ['primary_key' => ['roleid'], 'foreign_key' => [], 'operator_str' => db_prefix().'staff.role = '.db_prefix().'roles.roleid'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => [], 'operator_str' => db_prefix().'staff.team_manage = '.db_prefix().'staff.staffid'],

	]];


	$tables[]=['name' => 'roles', 'label' => _l('tblroles'), 'value' => [
		'roles' => ['primary_key' => ['roleid'], 'foreign_key' => []],

	]];

	$tables[]= ['name' => 'staff_departments', 'label' => _l('tblstaff_departments'), 'value' => [
		'staff_departments' => ['primary_key' => ['staffdepartmentid'], 'foreign_key' => ['staffid', 'departmentid']],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'staff_departments.staffid = '.db_prefix().'staff.staffid'],
		'departments' => ['primary_key' => ['departmentid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'staff_departments.departmentid = '.db_prefix().'departments.departmentid'],
	]];

	//departments
	$tables[]=['name' => 'departments', 'label' => _l('tbldepartments'), 'value' => [
		'departments' => ['primary_key' => ['departmentid'], 'foreign_key' => []],

	]];

	//clients
	$tables[]=['name' => 'clients', 'label' => _l('tblclients'), 'value' => [
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency']],

		'countries' => ['primary_key' => ['country_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'clients.country = '.db_prefix().'countries.country_id'],
		'leads' => ['primary_key' => ['id'], 'foreign_key' => ['assigned', 'from_form_id', 'status', 'addedfrom', 'client_id'], 'operator_str' => db_prefix().'clients.leadid = '.db_prefix().'leads.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'clients.addedfrom = '.db_prefix().'staff.staffid'],
		'currencies' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'clients.default_currency = '.db_prefix().'currencies.id'],

	]];

	//countries
	$tables[]=['name' => 'countries', 'label' => _l('tblcountries'), 'value' => [
		'countries' => ['primary_key' => ['country_id'], 'foreign_key' => []],
	]];

	//contacts
	$tables[]=['name' => 'contacts', 'label' => _l('tblcontacts'), 'value' => [
		'contacts' => ['primary_key' => ['id'], 'foreign_key' => ['userid']],

		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'contacts.userid = '.db_prefix().'clients.userid'],

	]];

	//customer_groups
	$tables[]=['name' => 'customer_groups', 'label' => _l('tblcustomer_groups'),  'value' => [
		'customer_groups' => ['primary_key' => ['id'], 'foreign_key' => ['groupid', 'customer_id']],

		'customers_groups' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'customer_groups.groupid = '.db_prefix().'customers_groups.id'],
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'customer_groups.customer_id = '.db_prefix().'clients.userid'],

	]];

	//Invoice
	$tables[]=['name' => 'invoices', 'label' => _l('tblinvoices'), 'value' => [
		'invoices' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'addedfrom', 'status', 'sale_agent', 'project_id', 'subscription_id' ]],
		
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'invoices.clientid = '.db_prefix().'clients.userid'],
		'currencies' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'invoices.currency = '.db_prefix().'currencies.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'invoices.addedfrom = '.db_prefix().'staff.staffid', 1 => db_prefix().'invoices.sale_agent = '.db_prefix().'staff.staffid'] ],
		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'invoices.project_id = '.db_prefix().'projects.id'],
		'subscriptions' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'tax_id', 'project_id', 'created_from'], 'operator_str' => db_prefix().'invoices.subscription_id = '.db_prefix().'subscriptions.id'],

	]];

	//estimates
	$tables[]=['name' => 'estimates', 'label' => _l('tblestimates'), 'value' => [
		'estimates' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'project_id', 'currency', 'addedfrom', 'status', 'sale_agent' ]],
		
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'estimates.clientid = '.db_prefix().'clients.userid'],
		'currencies' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'estimates.currency = '.db_prefix().'currencies.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'estimates.addedfrom = '.db_prefix().'staff.staffid', 1 => db_prefix().'estimates.sale_agent = '.db_prefix().'staff.staffid'] ],
		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'estimates.project_id = '.db_prefix().'projects.id'],

	]];

	//creditnotes
	$tables[]=['name' => 'creditnotes', 'label' => _l('tblcreditnotes'), 'value' => [
		'creditnotes' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'project_id', 'currency', 'addedfrom', 'status' ]],
		
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'creditnotes.clientid = '.db_prefix().'clients.userid'],
		'currencies' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'creditnotes.currency = '.db_prefix().'currencies.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'creditnotes.addedfrom = '.db_prefix().'staff.staffid'],
		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'creditnotes.project_id = '.db_prefix().'projects.id'],

	]];

	//tickets
	$tables[]=['name' => 'tickets', 'label' => _l('tbltickets'), 'value' => [
		'tickets' => ['primary_key' => ['ticketid'], 'foreign_key' => ['adminreplying', 'userid', 'contactid', 'department', 'status', 'project_id', 'assigned' ]],
		
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [ 0 => db_prefix().'tickets.adminreplying = '.db_prefix().'staff.staffid', 1 => db_prefix().'tickets.assigned = '.db_prefix().'staff.staffid']],
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'tickets.userid = '.db_prefix().'clients.userid'],
		'contacts' => ['primary_key' => ['id'], 'foreign_key' => ['userid'], 'operator_str' => db_prefix().'tickets.contactid = '.db_prefix().'contacts.id'],
		'departments' => ['primary_key' => ['departmentid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'tickets.department = '.db_prefix().'departments.departmentid'],
		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'tickets.project_id = '.db_prefix().'projects.id'],

	]];

	//subscriptions
	$tables[]=['name' => 'subscriptions', 'label' => _l('tblsubscriptions'), 'value' => [
		'subscriptions' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'tax_id', 'project_id', 'created_from' ]],
		
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'subscriptions.clientid = '.db_prefix().'clients.userid'],
		'currencies' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'subscriptions.currency = '.db_prefix().'currencies.id'],
		'taxes' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'subscriptions.tax_id = '.db_prefix().'taxes.id'],
		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'subscriptions.project_id = '.db_prefix().'projects.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'subscriptions.created_from = '.db_prefix().'staff.staffid'],

	]];

	//contracts
	$tables[]=['name' => 'contracts', 'label' => _l('tblcontracts'), 'value' => [
		'contracts' => ['primary_key' => ['id'], 'foreign_key' => ['client', 'contract_type', 'project_id', 'addedfrom' ]],
		
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'contracts.client = '.db_prefix().'clients.userid'],
		'contracts_types' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'contracts.contract_type = '.db_prefix().'contracts_types.id'],
		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'contracts.project_id = '.db_prefix().'projects.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'contracts.addedfrom = '.db_prefix().'staff.staffid'],
		
	]];

	//tasks
	$tables[]=['name' => 'tasks', 'label' => _l('tbltasks'), 'value' => [
		'tasks' => ['primary_key' => ['id'], 'foreign_key' => ['addedfrom', 'status', 'rel_id', 'rel_type', 'invoice_id' ]],
		
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'tasks.addedfrom = '.db_prefix().'staff.staffid'],
		'invoices' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'addedfrom', 'status', 'sale_agent', 'project_id', 'subscription_id'], 'operator_str' => db_prefix().'tasks.invoice_id = '.db_prefix().'invoices.id'],
		
	]];

	//task_comments
	$tables[]=['name' => 'task_comments', 'label' => _l('tbltask_comments'), 'value' => [
		'task_comments' => ['primary_key' => ['id'], 'foreign_key' => ['taskid', 'staffid', 'contact_id', 'file_id' ]],
		
		'tasks' => ['primary_key' => ['id'], 'foreign_key' => ['addedfrom', 'status', 'rel_id', 'rel_type', 'invoice_id' ], 'operator_str' => db_prefix().'task_comments.taskid = '.db_prefix().'tasks.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'task_comments.staffid = '.db_prefix().'staff.staffid'],
		'contacts' => ['primary_key' => ['id'], 'foreign_key' => ['userid'], 'operator_str' => db_prefix().'task_comments.contact_id = '.db_prefix().'contacts.id'],
		'files' => ['primary_key' => ['id'], 'foreign_key' => ['rel_id', 'rel_type', 'staffid', 'contact_id', 'task_comment_id'], 'operator_str' => db_prefix().'task_comments.invoice_id = '.db_prefix().'files.id'],
		
	]];

	//reminders
	$tables[]=['name' => 'reminders', 'label' => _l('tblreminders'), 'value' => [
		'reminders' => ['primary_key' => ['id'], 'foreign_key' => ['rel_id', 'staff', 'creator' ]],
		
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [ 0 => db_prefix().'reminders.staff = '.db_prefix().'staff.staffid', 1 => db_prefix().'reminders.creator = '.db_prefix().'staff.staffid']],
		
	]];

	//proposals
	$tables[]=['name' => 'proposals', 'label' => _l('tblproposals'), 'value' => [
		'proposals' => ['primary_key' => ['id'], 'foreign_key' => ['addedfrom', 'assigned', 'country', 'status','estimate_id', 'invoice_id']],
		
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [ 0 => db_prefix().'proposals.addedfrom = '.db_prefix().'staff.staffid', 1 => db_prefix().'proposals.assigned = '.db_prefix().'staff.staffid']],
		'countries' => ['primary_key' => ['country_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'proposals.country = '.db_prefix().'countries.country_id'],
		'estimates' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'project_id', 'currency', 'addedfrom', 'status', 'sale_agent'], 'operator_str' => db_prefix().'proposals.estimate_id = '.db_prefix().'estimates.id'],
		'invoices' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'addedfrom', 'status', 'sale_agent', 'project_id', 'subscription_id'], 'operator_str' => db_prefix().'proposals.invoice_id = '.db_prefix().'invoices.id'],

	]];

	//proposal_comments
	$tables[]=['name' => 'eads', 'label' => _l('tblproposal_comments'), 'value' => [
		'proposal_comments' => ['primary_key' => ['id'], 'foreign_key' => ['proposalid', 'staffid']],
		
		'proposals' => ['primary_key' => ['id'], 'foreign_key' => ['addedfrom', 'assigned', 'country', 'status','estimate_id', 'invoice_id'], 'operator_str' => db_prefix().'proposal_comments.proposalid = '.db_prefix().'proposals.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'proposal_comments.staffid = '.db_prefix().'staff.staffid'],
		
	]];

	//expenses
	$tables[]=['name' => 'expenses', 'label' => _l('tblexpenses'), 'value' => [
		'expenses' => ['primary_key' => ['id'], 'foreign_key' => ['category', 'currency', 'clientid', 'invoiceid', 'project_id', 'addedfrom']],
		
		'currencies' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'expenses.currency = '.db_prefix().'currencies.id'],
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'expenses.clientid = '.db_prefix().'clients.userid'],
		'invoices' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'addedfrom', 'status', 'sale_agent', 'project_id', 'subscription_id'], 'operator_str' => db_prefix().'expenses.invoiceid = '.db_prefix().'invoices.id'],
		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'expenses.project_id = '.db_prefix().'projects.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'expenses.addedfrom = '.db_prefix().'staff.staffid'],
		
	]];

	//leads
	$tables[]=['name' => 'leads', 'label' => _l('tblleads'), 'value' => [
		'leads' => ['primary_key' => ['id'], 'foreign_key' => ['country', 'assigned', 'from_form_id', 'status','source', 'addedfrom', 'client_id']],
		
		'countries' => ['primary_key' => ['country_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'leads.country = '.db_prefix().'countries.country_id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'leads.assigned = '.db_prefix().'staff.staffid', 1=> db_prefix().'leads.addedfrom = '.db_prefix().'staff.staffid']],
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'leads.client_id = '.db_prefix().'clients.userid'],

	]];

	//projects
	$tables[]=['name' => 'projects', 'label' => _l('tblprojects'), 'value' => [
		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification']],
		
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'projects.clientid = '.db_prefix().'clients.userid'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'projects.addedfrom = '.db_prefix().'staff.staffid'],

	]];

	//projectdiscussions
	$tables[]=['name' => 'projectdiscussions', 'label' => _l('tblprojectdiscussions'), 'value' => [
		'projectdiscussions' => ['primary_key' => ['id'], 'foreign_key' => ['project_id', 'staff_id', 'contact_id']],
		
		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'projectdiscussions.project_id = '.db_prefix().'projects.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'projectdiscussions.staff_id = '.db_prefix().'staff.staffid'],
		'contacts' => ['primary_key' => ['id'], 'foreign_key' => ['userid'], 'operator_str' => db_prefix().'projectdiscussions.contact_id = '.db_prefix().'contacts.id'],

	]];

	//projectdiscussioncomments
	$tables[]=['name' => 'projectdiscussioncomments', 'label' => _l('tblprojectdiscussioncomments'), 'value' => [
		'projectdiscussioncomments' => ['primary_key' => ['id'], 'foreign_key' => ['discussion_id', 'staff_id', 'contact_id']],
		
		'projectdiscussions' => ['primary_key' => ['id'], 'foreign_key' => ['project_id', 'staff_id', 'contact_id'], 'operator_str' => db_prefix().'projectdiscussioncomments.discussion_id = '.db_prefix().'projectdiscussions.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'projectdiscussioncomments.staff_id = '.db_prefix().'staff.staffid'],
		'contacts' => ['primary_key' => ['id'], 'foreign_key' => ['userid'], 'operator_str' => db_prefix().'projectdiscussioncomments.contact_id = '.db_prefix().'contacts.id'],
		
	]];

	//item_tax
	$tables[]=['name' => 'item_tax', 'label' => _l('tblitem_tax'), 'value' => [
		'item_tax' => ['primary_key' => ['id'], 'foreign_key' => ['itemid', 'rel_id']],
		
		'items' => ['primary_key' => ['id'], 'foreign_key' => ['tax', 'tax2', 'group_id', 'color_id', 'style_id', 'model_id', 'size_id', 'unit_id', 'sub_group'], 'operator_str' => db_prefix().'item_tax.itemid = '.db_prefix().'items.id'],
		
	]];

	//itemable TODO ( if rel_type, rel_id => need handle)
	$tables[]=['name' => 'itemable', 'label' => _l('tblitemable'), 'value' => [
		'itemable' => ['primary_key' => ['id'], 'foreign_key' => ['rel_id', 'rel_type']],
		
	]];

	//currencies
	$tables[]=['name' => 'currencies', 'label' => _l('tblcurrencies'), 'value' => [
		'currencies' => ['primary_key' => ['id'], 'foreign_key' => []],
		
	]];

	//notes TODO
	$tables[]=['name' => 'notes', 'label' => _l('tblnotes'), 'value' => [
		'notes' => ['primary_key' => ['id'], 'foreign_key' => ['rel_id', 'rel_type', 'addedfrom']],
		
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'notes.addedfrom = '.db_prefix().'staff.staffid'],

	]];

	//invoicepaymentrecords 
	$tables[]=['name' => 'invoicepaymentrecords', 'label' => _l('tblinvoicepaymentrecords'), 'value' => [
		'invoicepaymentrecords' => ['primary_key' => ['id'], 'foreign_key' => ['invoiceid']],
		
		'invoices' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'addedfrom', 'status', 'sale_agent', 'project_id', 'subscription_id'], 'operator_str' => db_prefix().'invoicepaymentrecords.invoiceid = '.db_prefix().'invoices.id'],

	]];

	//credits 
	$tables[]=['name' => 'credits', 'label' => _l('tblcredits'), 'value' => [
		'credits' => ['primary_key' => ['id'], 'foreign_key' => ['invoice_id', 'credit_id', 'staff_id']],
		
		'invoices' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'addedfrom', 'status', 'sale_agent', 'project_id', 'subscription_id'], 'operator_str' => db_prefix().'credits.invoice_id = '.db_prefix().'invoices.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'credits.staff_id = '.db_prefix().'staff.staffid'],


	]];
	

	//creditnotes 
	$tables[]=['name' => 'creditnotes', 'label' => _l('tblcreditnotes'), 'value' => [
		'creditnotes' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'addedfrom', 'status', 'project_id']],
		
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'creditnotes.clientid = '.db_prefix().'clients.userid'],
		'currencies' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'creditnotes.currency = '.db_prefix().'currencies.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'creditnotes.addedfrom = '.db_prefix().'staff.staffid'],
		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'creditnotes.project_id = '.db_prefix().'projects.id'],

	]];
	

	//related_items  TODO
	$tables[]=['name' => 'related_items', 'label' => _l('tblrelated_items'), 'value' => [
		'related_items' => ['primary_key' => ['id'], 'foreign_key' => ['rel_id', 'rel_type', 'item_id']],
		
		'items' => ['primary_key' => ['id'], 'foreign_key' => ['tax', 'tax2', 'group_id', 'color_id', 'style_id', 'model_id', 'size_id', 'unit_id', 'sub_group'], 'operator_str' => db_prefix().'related_items.item_id = '.db_prefix().'items.id'],

	]];
	

	//payment_modes 
	$tables[]=['name' => 'payment_modes', 'label' => _l('tblpayment_modes'), 'value' => [
		'payment_modes' => ['primary_key' => ['id'], 'foreign_key' => []],
		
	]];
	

	//creditnote_refunds 
	$tables[]=['name' => 'creditnote_refunds', 'label' => _l('tblcreditnote_refunds'), 'value' => [
		'creditnote_refunds' => ['primary_key' => ['id'], 'foreign_key' => ['credit_note_id', 'payment_mode', 'staff_id']],
		
		'creditnotes' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'addedfrom', 'status', 'project_id'], 'operator_str' => db_prefix().'creditnote_refunds.item_id = '.db_prefix().'creditnotes.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'creditnote_refunds.staff_id = '.db_prefix().'staff.staffid'],

	]];
	

	//items
	$tables[]=['name' => 'items', 'label' => _l('tblitems'), 'value' => [
		'items' => ['primary_key' => ['id'], 'foreign_key' => ['tax', 'tax2', 'group_id', 'color_id', 'style_id', 'model_id', 'size_id', 'unit_id', 'sub_group']],

		'taxes' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => [0 => db_prefix().'items.tax = '.db_prefix().'taxes.id', 1 => db_prefix().'items.tax2 = '.db_prefix().'taxes.id']],
		'items_groups' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'items.group_id = '.db_prefix().'items_groups.id'],
		'ware_color' => ['primary_key' => ['color_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'items.color_id = '.db_prefix().'ware_color.color_id'],
		'ware_style_type' => ['primary_key' => ['style_type_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'items.style_id = '.db_prefix().'ware_style_type.style_type_id'],
		'ware_body_type' => ['primary_key' => ['body_type_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'items.model_id = '.db_prefix().'ware_body_type.body_type_id'],
		'ware_size_type' => ['primary_key' => ['size_type_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'items.size_id = '.db_prefix().'ware_size_type.size_type_id'],
		'ware_unit_type' => ['primary_key' => ['unit_type_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'items.unit_id = '.db_prefix().'ware_unit_type.unit_type_id'],
		'wh_sub_group' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'items.sub_group = '.db_prefix().'wh_sub_group.id'],

	]];

	//items_groups
	$tables[]=['name' => 'items_groups', 'label' => _l('tblitems_groups'), 'value' => [
		'items_groups' => ['primary_key' => ['id'], 'foreign_key' => []],

	]];

	//taxes
	$tables[]=['name' => 'taxes', 'label' => _l('tbltaxes'), 'value' => [
		'taxes' => ['primary_key' => ['id'], 'foreign_key' => []],

	]];

	//expenses_categories
	$tables[]=['name' => 'expenses_categories', 'label' => _l('tblexpenses_categories'), 'value' => [
		'expenses_categories' => ['primary_key' => ['id'], 'foreign_key' => []],

	]];

	//contracts_types
	$tables[]=['name' => 'contracts_types', 'label' => _l('tblcontracts_types'), 'value' => [
		'contracts_types' => ['primary_key' => ['id'], 'foreign_key' => []],

	]];

	//contract_comments
	$tables[]=['name' => 'contract_comments', 'label' => _l('tblcontract_comments'), 'value' => [
		'contract_comments' => ['primary_key' => ['id'], 'foreign_key' => ['contract_id', 'staffid']],

		'contracts' => ['primary_key' => ['id'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'contract_comments.contract_id = '.db_prefix().'contracts.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'contract_comments.staffid = '.db_prefix().'staff.staffid'],

	]];

	//contract_renewals
	$tables[]=['name' => 'contract_renewals', 'label' => _l('tblcontract_renewals'), 'value' => [
		'contract_renewals' => ['primary_key' => ['id'], 'foreign_key' => ['contractid', 'renewed_by_staff_id']],

		'contracts' => ['primary_key' => ['id'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'contract_renewals.contractid = '.db_prefix().'contracts.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'contract_renewals.renewed_by_staff_id = '.db_prefix().'staff.staffid'],
	]];

	
	//taskstimers
	$tables[]=['name' => 'taskstimers', 'label' => _l('tbltaskstimers'), 'value' => [
		'taskstimers' => ['primary_key' => ['id'], 'foreign_key' => ['task_id', 'staff_id']],

		'tasks' => ['primary_key' => ['id'], 'foreign_key' => ['addedfrom', 'status', 'rel_id', 'rel_type', 'invoice_id' ], 'operator_str' => db_prefix().'taskstimers.task_id = '.db_prefix().'tasks.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'taskstimers.staff_id = '.db_prefix().'staff.staffid'],
	]];

	//project_members
	$tables[]=['name' => 'project_members', 'label' => _l('tblproject_members'), 'value' => [
		'project_members' => ['primary_key' => ['id'], 'foreign_key' => ['project_id', 'staff_id']],

		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'project_members.project_id = '.db_prefix().'projects.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'project_members.staff_id = '.db_prefix().'staff.staffid'],
	]];

	
	//pinned_projects
	$tables[]=['name' => 'pinned_projects', 'label' => _l('tblpinned_projects'), 'value' => [
		'pinned_projects' => ['primary_key' => ['id'], 'foreign_key' => ['project_id', 'staff_id']],

		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'pinned_projects.project_id = '.db_prefix().'projects.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'pinned_projects.staff_id = '.db_prefix().'staff.staffid'],
	]];

	//project_settings
	$tables[]=['name' => 'project_settings', 'label' => _l('tblproject_settings'), 'value' => [
		'project_settings' => ['primary_key' => ['id'], 'foreign_key' => ['project_id']],

		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'project_settings.project_id = '.db_prefix().'projects.id'],
	]];

	
	//milestones
	$tables[]=['name' => 'milestones', 'label' => _l('tblmilestones'), 'value' => [
		'milestones' => ['primary_key' => ['id'], 'foreign_key' => ['project_id']],

		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'milestones.project_id = '.db_prefix().'projects.id'],
	]];

	//task_assigned
	$tables[]=['name' => 'task_assigned', 'label' => _l('tbltask_assigned'), 'value' => [
		'task_assigned' => ['primary_key' => ['id'], 'foreign_key' => ['staffid', 'taskid', 'assigned_from']],

		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'task_assigned.staffid = '.db_prefix().'staff.staffid', 1=> db_prefix().'task_assigned.assigned_from = '.db_prefix().'staff.staffid']],
		'tasks' => ['primary_key' => ['id'], 'foreign_key' => ['addedfrom', 'status', 'rel_id', 'rel_type', 'invoice_id' ], 'operator_str' => db_prefix().'task_assigned.taskid = '.db_prefix().'tasks.id'],
	]];

	//task_followers
	$tables[]=['name' => 'task_followers', 'label' => _l('tbltask_followers'), 'value' => [
		'task_followers' => ['primary_key' => ['id'], 'foreign_key' => ['staffid', 'taskid']],

		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'task_followers.staffid = '.db_prefix().'staff.staffid'],
		'tasks' => ['primary_key' => ['id'], 'foreign_key' => ['addedfrom', 'status', 'rel_id', 'rel_type', 'invoice_id' ], 'operator_str' => db_prefix().'task_followers.taskid = '.db_prefix().'tasks.id'],
	]];

	//project_notes
	$tables[]=['name' => 'project_notes', 'label' => _l('tblproject_notes'), 'value' => [
		'project_notes' => ['primary_key' => ['id'], 'foreign_key' => ['project_id', 'staffid']],

		'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'project_notes.project_id = '.db_prefix().'projects.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'project_notes.staffid = '.db_prefix().'staff.staffid'],
	]];

	//task_checklist_items
	$tables[]=['name' => 'task_checklist_items', 'label' => _l('tbltask_checklist_items'), 'value' => [
		'task_checklist_items' => ['primary_key' => ['id'], 'foreign_key' => ['taskid', 'assigned']],

		'tasks' => ['primary_key' => ['id'], 'foreign_key' => ['addedfrom', 'status', 'rel_id', 'rel_type', 'invoice_id' ], 'operator_str' => db_prefix().'task_checklist_items.taskid = '.db_prefix().'tasks.id'],
		'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'task_checklist_items.assigned = '.db_prefix().'staff.staffid'],
	]];

	//ticket_replies
	$tables[]=['name' => 'ticket_replies', 'label' => _l('tblticket_replies'), 'value' => [
		'ticket_replies' => ['primary_key' => ['id'], 'foreign_key' => ['ticketid', 'userid', 'contactid']],


		'tickets' => ['primary_key' => ['ticketid'], 'foreign_key' => ['adminreplying', 'userid', 'contactid', 'department', 'status', 'project_id', 'assigned'], 'operator_str' => db_prefix().'ticket_replies.userid = '.db_prefix().'tickets.ticketid'],
		'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'ticket_replies.userid = '.db_prefix().'clients.userid'],
		'contacts' => ['primary_key' => ['id'], 'foreign_key' => ['userid'], 'operator_str' => db_prefix().'ticket_replies.contactid = '.db_prefix().'contacts.id'],
	]];

	//tickets_pipe_log
	$tables[]=['name' => 'tickets_pipe_log', 'label' => _l('tbltickets_pipe_log'), 'value' => [
		'tickets_pipe_log' => ['primary_key' => ['id'], 'foreign_key' => ['ticketid', 'userid', 'contactid']],

	]];

	//tickets_priorities
	$tables[]=['name' => 'tickets_priorities', 'label' => _l('tbltickets_priorities'), 'value' => [
		'tickets_priorities' => ['primary_key' => ['priorityid'], 'foreign_key' => []],

	]];

	//services
	$tables[]=['name' => 'services', 'label' => _l('tblservices'), 'value' => [
		'services' => ['primary_key' => ['serviceid'], 'foreign_key' => []],

	]];

	//tickets_status
	$tables[]=['name' => 'tickets_status', 'label' => _l('tbltickets_status'), 'value' => [
		'tickets_status' => ['primary_key' => ['ticketstatusid'], 'foreign_key' => []],

	]];

	//ticket_attachments
	$tables[]=['name' => 'ticket_attachments', 'label' => _l('tblticket_attachments'), 'value' => [
		'ticket_attachments' => ['primary_key' => ['ticketstatusid'], 'foreign_key' => ['ticketid', 'replyid']],

		'tickets' => ['primary_key' => ['ticketid'], 'foreign_key' => ['adminreplying', 'userid', 'contactid', 'department', 'status', 'project_id', 'assigned'], 'operator_str' => db_prefix().'ticket_attachments.ticketid = '.db_prefix().'tickets.ticketid'],

	]];

	//tickets_predefined_replies
	$tables[]=['name' => 'tickets_predefined_replies', 'label' => _l('tbltickets_predefined_replies'), 'value' => [
		'tickets_predefined_replies' => ['primary_key' => ['id'], 'foreign_key' => []],

	]];

	//leads_status
	$tables[]=['name' => 'leads_status', 'label' => _l('tblleads_status'), 'value' => [
		'leads_status' => ['primary_key' => ['id'], 'foreign_key' => []],

	]];

	//leads_sources
	$tables[]=['name' => 'leads_sources', 'label' => _l('tblleads_sources'), 'value' => [
		'leads_sources' => ['primary_key' => ['id'], 'foreign_key' => []],

	]];

	//leads_email_integration
	$tables[]=['name' => 'leads_email_integration', 'label' => _l('tblleads_email_integration'), 'value' => [
		'leads_email_integration' => ['primary_key' => ['id'], 'foreign_key' => []],

	]];

	//web_to_lead
	$tables[]=['name' => 'web_to_lead', 'label' => _l('tblweb_to_lead'), 'value' => [
		'web_to_lead' => ['primary_key' => ['id'], 'foreign_key' => ['lead_source', 'lead_status']],

		'leads_sources' => ['id' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'web_to_lead.lead_source = '.db_prefix().'leads_sources.id'],
		'tblleads_status' => ['id' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'web_to_lead.lead_status = '.db_prefix().'tblleads_status.id'],

	]];
	

	//warehouse module
	if(rb_get_status_modules('warehouse')){
		//lost adjustment table
		$tables[]=['name' => 'wh_loss_adjustment', 'label' => _l('tblwh_loss_adjustment'), 'value' => [
			'wh_loss_adjustment' => ['primary_key' => ['id'], 'foreign_key' => ['warehouses']],

			'warehouse' => ['primary_key' => ['warehouse_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'wh_loss_adjustment.warehouses = '.db_prefix().'warehouse.warehouse_id'],

		]];

		//wh_loss_adjustment_detail
		$tables[]=['name' => 'wh_loss_adjustment_detail', 'label' => _l('tblwh_loss_adjustment_detail'), 'value' => [
			'wh_loss_adjustment_detail' => ['primary_key' => ['id'], 'foreign_key' => ['loss_adjustment', 'items', 'unit']],

			'wh_loss_adjustment' => ['primary_key' => ['id'], 'foreign_key' => ['warehouses'], 'operator_str' => db_prefix().'wh_loss_adjustment_detail.loss_adjustment = '.db_prefix().'wh_loss_adjustment.id'],
			'items' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'wh_loss_adjustment_detail.items = '.db_prefix().'items.id'],
			'ware_unit_type' => ['primary_key' => ['unit_type_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'wh_loss_adjustment_detail.unit = '.db_prefix().'ware_unit_type.unit_type_id'],

		]];

		

		//goods receipt table
		$tables[]=['name' => 'goods_receipt', 'label' => _l('tblgoods_receipt'), 'value' => [
			'goods_receipt' => ['primary_key' => ['id'], 'foreign_key' => ['buyer_id', 'pr_order_id']],

			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'goods_receipt.buyer_id = '.db_prefix().'staff.staffid'],
			'pur_orders' => ['primary_key' => ['id'], 'foreign_key' => ['buyer'], 'operator_str' => db_prefix().'goods_receipt.pr_order_id = '.db_prefix().'pur_orders.id'],

		]];

		//goods receipt detail table
		$tables[]=['name' => 'goods_receipt_detail', 'label' => _l('tblgoods_receipt_detail'), 'value' => [
			'goods_receipt_detail' => ['primary_key' => ['id'], 'foreign_key' => ['goods_receipt_id', 'commodity_code', 'warehouse_id', 'unit_id']],
			
			'goods_receipt' => ['primary_key' => ['id'], 'foreign_key' => ['buyer_id', 'warehouse_id', 'pr_order_id'], 'operator_str' => db_prefix().'goods_receipt_detail.goods_receipt_id = '.db_prefix().'goods_receipt.id'],
			'pur_orders' => ['primary_key' => ['id'], 'foreign_key' => ['buyer'], 'operator_str' => db_prefix().'goods_receipt.pr_order_id = '.db_prefix().'pur_orders.id'],
			'items' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'goods_receipt_detail.commodity_code = '.db_prefix().'items.id'],
			'warehouse' => ['primary_key' => ['warehouse_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'goods_receipt_detail.warehouse_id = '.db_prefix().'warehouse.warehouse_id'],
			'ware_unit_type' => ['primary_key' => ['unit_type_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'goods_receipt_detail.unit_id = '.db_prefix().'ware_unit_type.unit_type_id'],

		]];


		//goods delivery table
		$tables[]=['name' => 'goods_delivery', 'label' => _l('tblgoods_delivery'), 'value' => [
			'goods_delivery' => ['primary_key' => ['id'], 'foreign_key' => ['staff_id', 'invoice_id', 'pr_order_id', 'addedfrom', 'customer_code', 'project', 'department', 'requester']],

			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'goods_delivery.staff_id = '.db_prefix().'staff.staffid', 1 => db_prefix().'goods_delivery.addedfrom = '.db_prefix().'staff.staffid', 2 => db_prefix().'goods_delivery.requester = '.db_prefix().'staff.staffid']],
			'pur_orders' => ['primary_key' => ['id'], 'foreign_key' => ['buyer'], 'operator_str' => db_prefix().'goods_delivery.pr_order_id = '.db_prefix().'pur_orders.id'],
			'invoices' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'sale_agent', 'project_id'], 'operator_str' => db_prefix().'goods_delivery.invoice_id = '.db_prefix().'invoices.id'],
			'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'goods_delivery.customer_code = '.db_prefix().'clients.userid'],
			'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'goods_delivery.project = '.db_prefix().'projects.id'],
			'departments' => ['primary_key' => ['departmentid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'goods_delivery.department = '.db_prefix().'departments.departmentid'],



		]];

		//goods delivery  detail table
		$tables[]=['name' => 'goods_delivery_detail', 'label' => _l('tblgoods_delivery_detail'), 'value' => [
			'goods_delivery_detail' => ['primary_key' => ['id'], 'foreign_key' => ['goods_delivery_id', 'commodity_code', 'warehouse_id', 'unit_id']],
			
			'goods_delivery' => ['primary_key' => ['id'], 'foreign_key' => ['buyer_id', 'warehouse_id', 'pr_order_id'], 'operator_str' => db_prefix().'goods_receipt_detail.goods_delivery_id = '.db_prefix().'goods_delivery.id'],
			'items' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'goods_delivery_detail.commodity_code = '.db_prefix().'items.id'], 
			'warehouse' => ['primary_key' => ['warehouse_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'goods_delivery_detail.warehouse_id = '.db_prefix().'warehouse.warehouse_id'],
			'ware_unit_type' => ['primary_key' => ['unit_type_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'goods_delivery_detail.unit_id = '.db_prefix().'ware_unit_type.unit_type_id'],

		]];

		//internal transfer
		$tables[]=['name' => 'internal_delivery_note', 'label' => _l('tblinternal_delivery_note'), 'value' => [
			'internal_delivery_note' => ['primary_key' => ['id'], 'foreign_key' => ['staff_id', 'addedfrom']],

			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'internal_delivery_note.staff_id = '.db_prefix().'staff.staffid', 1 => db_prefix().'internal_delivery_note.addedfrom = '.db_prefix().'staff.staffid']],

		]];


		//internal transfer details
		$tables[]=['name' => 'internal_delivery_note_detail', 'label' => _l('tblinternal_delivery_note_detail'), 'value' => [
			'internal_delivery_note_detail' => ['primary_key' => ['id'], 'foreign_key' => ['internal_delivery_id', 'commodity_code', 'from_stock_name', 'to_stock_name', 'unit_id']],

			'internal_delivery_note' => ['primary_key' => ['id'], 'foreign_key' => ['staff_id', 'addedfrom'], 'operator_str' => db_prefix().'internal_delivery_note_detail.internal_delivery_id = '.db_prefix().'internal_delivery_note.id'],
			'items' => ['primary_key' => ['id'], 'foreign_key' => ['tax', 'tax2', 'group_id', 'color_id', 'style_id', 'model_id', 'size_id', 'unit_id', 'sub_group'], 'operator_str' => db_prefix().'internal_delivery_note_detail.commodity_code = '.db_prefix().'items.id'],
			'warehouse' => ['primary_key' => ['warehouse_id'], 'foreign_key' => [], 'operator_str' => [0 => db_prefix().'internal_delivery_note_detail.from_stock_name = '.db_prefix().'warehouse.warehouse_id', 1 => db_prefix().'internal_delivery_note_detail.to_stock_name = '.db_prefix().'warehouse.warehouse_id' ] ],
			'ware_unit_type' => ['primary_key' => ['unit_type_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'internal_delivery_note_detail.unit_id = '.db_prefix().'ware_unit_type.unit_type_id'],

		]];

		//warehouse
		$tables[]=['name' => 'warehouse', 'label' => _l('tblwarehouse'), 'value' => [
			'warehouse' => ['primary_key' => ['warehouse_id'], 'foreign_key' => ['country']],

			'countries' => ['primary_key' => ['country_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'warehouse.country = '.db_prefix().'countries.country_id'],

		]];

		//inventory_manage
		$tables[]=['name' => 'inventory_manage', 'label' => _l('tblinventory_manage'), 'value' => [
			'inventory_manage' => ['primary_key' => ['id'], 'foreign_key' => ['warehouse_id', 'commodity_id']],

			'warehouse' => ['primary_key' => ['warehouse_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'inventory_manage.warehouse_id = '.db_prefix().'warehouse.warehouse_id'],
			'items' => ['primary_key' => ['id'], 'foreign_key' => ['tax', 'tax2', 'group_id', 'color_id', 'style_id', 'model_id', 'size_id', 'unit_id', 'sub_group'], 'operator_str' => db_prefix().'inventory_manage.commodity_id = '.db_prefix().'items.id'],
		]];

		

		//dont get goods_transaction_detail table

		// setting
		//Commodity type
		$tables[]=['name' => 'ware_commodity_type', 'label' => _l('tblware_commodity_type'), 'value' => [
			'ware_commodity_type' => ['primary_key' => ['commodity_type_id'], 'foreign_key' => []],

		]];

		//sub group
		$tables[]=['name' => 'wh_sub_group', 'label' => _l('tblwh_sub_group'), 'value' => [
			'wh_sub_group' => ['primary_key' => ['id'], 'foreign_key' => ['group_id']],
			
			'items_groups' => ['primary_key' => ['country_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'wh_sub_group.group_id = '.db_prefix().'items_groups.id'],

		]];
		
		
		//Unit
		$tables[]=['name' => 'ware_unit_type', 'label' => _l('tblware_unit_type'), 'value' => [
			'ware_unit_type' => ['primary_key' => ['unit_type_id'], 'foreign_key' => []],
			
		]];
		
		//Colors
		$tables[]=['name' => 'ware_color', 'label' => _l('tblware_color'), 'value' => [
			'ware_color' => ['primary_key' => ['color_id'], 'foreign_key' => []],

		]];
		
		//Models
		$tables[]=['name' => 'ware_body_type', 'label' => _l('tblware_body_type'), 'value' => [
			'ware_body_type' => ['primary_key' => ['body_type_id'], 'foreign_key' => []],

		]];
		
		//sizes
		$tables[]=['name' => 'ware_size_type', 'label' => _l('tblware_size_type'), 'value' => [
			'ware_size_type' => ['primary_key' => ['size_type_id'], 'foreign_key' => []],

		]];
		
		//Styles
		$tables[]=['name' => 'ware_style_type', 'label' => _l('tblware_style_type'), 'value' => [
			'ware_style_type' => ['primary_key' => ['style_type_id'], 'foreign_key' => []],

		]];
		
		 

	}

	// Purchase moudule
	if(rb_get_status_modules('purchase')){

		// pur_comments
		$tables[]=['name' => 'pur_comments', 'label' => _l('tblpur_comments'), 'value' => [
			'pur_comments' => ['primary_key' => ['id'], 'foreign_key' => ['staffid', 'rel_id', 'rel_type']],
			
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'pur_comments.staffid = '.db_prefix().'staff.staffid'],

		]];

		// pur_contacts
		$tables[]=['name' => 'pur_contacts', 'label' => _l('tblpur_contacts'), 'value' => [
			'pur_contacts' => ['primary_key' => ['id'], 'foreign_key' => ['userid']],
			
			'pur_vendor' => ['primary_key' => ['userid'], 'foreign_key' => [], 'operator_str' => db_prefix().'pur_contacts.userid = '.db_prefix().'pur_vendor.userid'],

		]];

		// pur_contracts
		$tables[]=['name' => 'pur_contracts', 'label' => _l('tblpur_contracts'), 'value' => [
			'pur_contracts' => ['primary_key' => ['id'], 'foreign_key' => ['vendor', 'pur_order', 'buyer', 'add_from', 'signer', 'project', 'department']],
			
			'pur_vendor' => ['primary_key' => ['userid'], 'foreign_key' => ['addedfrom', 'category'], 'operator_str' => db_prefix().'pur_contracts.vendor = '.db_prefix().'pur_vendor.userid'],
			'pur_orders' => ['primary_key' => ['id'], 'foreign_key' => ['vendor', 'estimate', 'addedfrom', 'buyer', 'clients', 'project', 'pur_request', 'department', 'sale_invoice'], 'operator_str' => db_prefix().'pur_contracts.pur_order = '.db_prefix().'pur_orders.id'],

			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'pur_contracts.buyer = '.db_prefix().'staff.staffid', 1 => db_prefix().'pur_contracts.add_from = '.db_prefix().'staff.staffid', 2 => db_prefix().'pur_contracts.signer = '.db_prefix().'staff.staffid']],
			'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'pur_contracts.project = '.db_prefix().'projects.id'],
			'departments' => ['primary_key' => ['departmentid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'pur_contracts.department = '.db_prefix().'departments.departmentid'],


		]];


		// pur_debit_notes
		$tables[]=['name' => 'pur_debit_notes', 'label' => _l('tblpur_debit_notes'), 'value' => [
			'pur_debit_notes' => ['primary_key' => ['id'], 'foreign_key' => ['vendorid', 'addedfrom']],
			
			'pur_vendor' => ['primary_key' => ['userid'], 'foreign_key' => ['addedfrom', 'category'], 'operator_str' => db_prefix().'pur_debit_notes.vendorid = '.db_prefix().'pur_vendor.userid'],
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'pur_debit_notes.addedfrom = '.db_prefix().'staff.staffid'],

		]];

		// pur_debits
		$tables[]=['name' => 'pur_debits', 'label' => _l('tblpur_debits'), 'value' => [
			'pur_debits' => ['primary_key' => ['id'], 'foreign_key' => ['invoice_id', 'debit_id', 'staff_id']],
			
			'pur_invoices' => ['primary_key' => ['id'], 'foreign_key' => ['vendor', 'add_from', 'contract', 'pur_order'], 'operator_str' => db_prefix().'pur_debits.invoice_id = '.db_prefix().'pur_invoices.id'],
			'pur_debit_notes' => ['primary_key' => ['id'], 'foreign_key' => ['vendorid', 'addedfrom'], 'operator_str' => db_prefix().'pur_debits.debit_id = '.db_prefix().'pur_debit_notes.id'],
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'pur_debits.staff_id = '.db_prefix().'staff.staffid'],

		]];

		// pur_debits_refunds
		$tables[]=['name' => 'pur_debits_refunds', 'label' => _l('tblpur_debits_refunds'), 'value' => [
			'pur_debits_refunds' => ['primary_key' => ['id'], 'foreign_key' => ['debit_note_id', 'staff_id']],
			
			'pur_debit_notes' => ['primary_key' => ['id'], 'foreign_key' => ['vendorid', 'addedfrom'], 'operator_str' => db_prefix().'pur_debits_refunds.debit_note_id = '.db_prefix().'pur_debit_notes.id'],
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'pur_debits_refunds.staff_id = '.db_prefix().'staff.staffid'],

		]];


		// pur_estimates
		$tables[]=['name' => 'pur_estimates', 'label' => _l('tblpur_estimates'), 'value' => [
			'pur_estimates' => ['primary_key' => ['id'], 'foreign_key' => ['vendor', 'pur_request', 'addedfrom', 'buyer']],
			
			'pur_vendor' => ['primary_key' => ['userid'], 'foreign_key' => ['addedfrom', 'category'], 'operator_str' => db_prefix().'pur_estimates.vendor = '.db_prefix().'pur_vendor.userid'],
			'pur_request' => ['primary_key' => ['id'], 'foreign_key' => ['requester', 'department', 'project', 'sale_invoice'], 'operator_str' => db_prefix().'pur_estimates.pur_request = '.db_prefix().'pur_request.id'],
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'pur_estimates.buyer = '.db_prefix().'staff.staffid', 1 => db_prefix().'pur_estimates.buyer = '.db_prefix().'staff.staffid' ]],

		]];
 

		// pur_estimate_detail
		$tables[]=['name' => 'pur_estimate_detail', 'label' => _l('tblpur_estimate_detail'), 'value' => [
			'pur_estimate_detail' => ['primary_key' => ['id'], 'foreign_key' => ['pur_estimate', 'item_code', 'unit_id']],
			
			'pur_estimates' => ['primary_key' => ['id'], 'foreign_key' => ['vendor', 'pur_request', 'addedfrom', 'buyer'], 'operator_str' => db_prefix().'pur_estimate_detail.pur_estimate = '.db_prefix().'pur_estimates.id'],
			'ware_unit_type' => ['primary_key' => ['unit_type_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'pur_estimate_detail.unit_id = '.db_prefix().'ware_unit_type.unit_type_id'],
			'items' => ['primary_key' => ['id'], 'foreign_key' => ['tax', 'tax2', 'group_id', 'color_id', 'style_id', 'model_id', 'size_id', 'unit_id', 'sub_group'], 'operator_str' => db_prefix().'pur_estimate_detail.item_code = '.db_prefix().'items.id'],


		]];


		// pur_invoices
		$tables[]=['name' => 'pur_invoices', 'label' => _l('tblpur_invoices'), 'value' => [
			'pur_invoices' => ['primary_key' => ['id'], 'foreign_key' => ['vendor','add_from','contract','pur_order']],
			
			'pur_vendor' => ['primary_key' => ['userid'], 'foreign_key' => ['addedfrom', 'category'], 'operator_str' => db_prefix().'pur_invoices.vendor = '.db_prefix().'pur_vendor.userid'],
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'pur_invoices.add_from = '.db_prefix().'staff.staffid' ]],
			'pur_contracts' => ['primary_key' => ['id'], 'foreign_key' => ['vendor', 'pur_order', 'buyer', 'add_from', 'signer', 'project', 'department'], 'operator_str' => db_prefix().'pur_invoices.contract = '.db_prefix().'pur_contracts.id'],
			'pur_orders' => ['primary_key' => ['id'], 'foreign_key' => ['vendor', 'estimate', 'addedfrom', 'buyer', 'clients', 'project', 'pur_request', 'department', 'sale_invoice'], 'operator_str' => db_prefix().'pur_invoices.pur_order = '.db_prefix().'pur_orders.id'],

		]];


		// pur_invoice_payment
		$tables[]=['name' => 'pur_invoice_payment', 'label' => _l('tblpur_invoice_payment'), 'value' => [
			'pur_invoice_payment' => ['primary_key' => ['id'], 'foreign_key' => ['pur_invoice', 'requester' ]],
			
			'pur_invoices' => ['primary_key' => ['id'], 'foreign_key' => ['vendor', 'add_from', 'contract', 'pur_order'], 'operator_str' => db_prefix().'pur_invoice_payment.pur_invoice = '.db_prefix().'pur_invoices.id'],
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'pur_invoice_payment.requester = '.db_prefix().'staff.staffid' ]],
			

		]];


		// pur_orders
		$tables[]=['name' => 'pur_orders', 'label' => _l('tblpur_orders'), 'value' => [
			'pur_orders' => ['primary_key' => ['id'], 'foreign_key' => ['vendor', 'estimate', 'addedfrom', 'buyer', 'clients', 'project', 'pur_request', 'department', 'sale_invoice']],
			
			'pur_vendor' => ['primary_key' => ['userid'], 'foreign_key' => ['addedfrom', 'category'], 'operator_str' => db_prefix().'pur_orders.vendor = '.db_prefix().'pur_vendor.userid'],
			'pur_estimates' => ['primary_key' => ['id'], 'foreign_key' => ['vendor', 'pur_request', 'addedfrom', 'buyer'], 'operator_str' => db_prefix().'pur_orders.estimate = '.db_prefix().'pur_estimates.id'],
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'pur_orders.buyer = '.db_prefix().'staff.staffid', 1 => db_prefix().'pur_orders.buyer = '.db_prefix().'staff.staffid' ]],
			'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'pur_orders.clients = '.db_prefix().'clients.userid'],
			'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'pur_orders.project = '.db_prefix().'projects.id'],
			'pur_request' => ['primary_key' => ['id'], 'foreign_key' => ['requester', 'department', 'project', 'sale_invoice'], 'operator_str' => db_prefix().'pur_orders.pur_request = '.db_prefix().'pur_request.id'],
			'departments' => ['primary_key' => ['departmentid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'pur_orders.department = '.db_prefix().'departments.departmentid'],
			'invoices' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'sale_agent', 'project_id'], 'operator_str' => db_prefix().'pur_orders.sale_invoice = '.db_prefix().'invoices.id'],


		]];


		// pur_order_detail
		$tables[]=['name' => 'pur_order_detail', 'label' => _l('tblpur_order_detail'), 'value' => [
			'pur_order_detail' => ['primary_key' => ['id'], 'foreign_key' => ['pur_order','item_code','unit_id']],

			'pur_orders' => ['primary_key' => ['id'], 'foreign_key' => ['vendor', 'estimate', 'addedfrom', 'buyer', 'clients', 'project', 'pur_request', 'department', 'sale_invoice'], 'operator_str' => db_prefix().'pur_order_detail.pur_order = '.db_prefix().'pur_orders.id'],
			'items' => ['primary_key' => ['id'], 'foreign_key' => ['tax', 'tax2', 'group_id', 'color_id', 'style_id', 'model_id', 'size_id', 'unit_id', 'sub_group'], 'operator_str' => db_prefix().'pur_order_detail.item_code = '.db_prefix().'items.id'],
			'ware_unit_type' => ['primary_key' => ['unit_type_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'pur_order_detail.unit_id = '.db_prefix().'ware_unit_type.unit_type_id'],


		]];



		// pur_request
		$tables[]=['name' => 'pur_request', 'label' => _l('tblpur_request'), 'value' => [
			'pur_request' => ['primary_key' => ['id'], 'foreign_key' => ['requester','department','project','sale_invoice' ]],
			
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'pur_request.requester = '.db_prefix().'staff.staffid' ]],
			'departments' => ['primary_key' => ['departmentid'], 'foreign_key' => ['role'], 'operator_str' => db_prefix().'pur_request.department = '.db_prefix().'departments.departmentid'],
			'projects' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'addedfrom', 'contact_notification'], 'operator_str' => db_prefix().'pur_request.project = '.db_prefix().'projects.id'],
			'invoices' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'sale_agent', 'project_id'], 'operator_str' => db_prefix().'pur_request.sale_invoice = '.db_prefix().'invoices.id'],


		]];


		// pur_request_detail
		$tables[]=['name' => 'pur_request_detail', 'label' => _l('tblpur_request_detail'), 'value' => [
			'pur_request_detail' => ['primary_key' => ['prd_id'], 'foreign_key' => ['pur_request','item_code','unit_id']],
			
			'pur_request' => ['primary_key' => ['id'], 'foreign_key' => ['requester', 'department', 'project', 'sale_invoice'], 'operator_str' => db_prefix().'pur_request_detail.pur_request = '.db_prefix().'pur_request.id'],
			'items' => ['primary_key' => ['id'], 'foreign_key' => ['tax', 'tax2', 'group_id', 'color_id', 'style_id', 'model_id', 'size_id', 'unit_id', 'sub_group'], 'operator_str' => db_prefix().'pur_request_detail.item_code = '.db_prefix().'items.id'],
			'ware_unit_type' => ['primary_key' => ['unit_type_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'pur_request_detail.unit_id = '.db_prefix().'ware_unit_type.unit_type_id'],


		]];


		// pur_vendor
		$tables[]=['name' => 'pur_vendor', 'label' => _l('tblpur_vendor'), 'value' => [
			'pur_vendor' => ['primary_key' => ['id'], 'foreign_key' => ['addedfrom','category',]],
			
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'pur_vendor.addedfrom = '.db_prefix().'staff.staffid' ]],
			'pur_vendor_cate' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => [0 => db_prefix().'pur_vendor.category = '.db_prefix().'pur_vendor_cate.id' ]],


		]];


		// pur_vendor_admin
		$tables[]=['name' => 'pur_vendor_admin', 'label' => _l('tblpur_vendor_admin'), 'value' => [
			'pur_vendor_admin' => ['primary_key' => ['id'], 'foreign_key' => ['staff_id', 'vendor_id' ]],
			
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'pur_vendor_admin.staff_id = '.db_prefix().'staff.staffid' ]],
			'pur_vendor' => ['primary_key' => ['userid'], 'foreign_key' => ['addedfrom', 'category'], 'operator_str' => db_prefix().'pur_vendor_admin.vendor_id = '.db_prefix().'pur_vendor.userid'],


		]];

		// pur_vendor_cate
		$tables[]=['name' => 'pur_vendor_cate', 'label' => _l('tblpur_vendor_cate'), 'value' => [
			'pur_vendor_cate' => ['primary_key' => ['id'], 'foreign_key' => []],
			
		]];

		

		

	}

	// Omnisalse
	if(rb_get_status_modules('omni_sales')){

		// sales_channel
		$tables[]=['name' => 'sales_channel', 'label' => _l('tblsales_channel'), 'value' => [
			'sales_channel' => ['primary_key' => ['id'], 'foreign_key' => [ ]],
			
		]];

		// sales_channel_detailt
		$tables[]=['name' => 'sales_channel_detailt', 'label' => _l('tblsales_channel_detailt'), 'value' => [
			'sales_channel_detailt' => ['primary_key' => ['id'], 'foreign_key' => ['group_product_id', 'product_id', 'department', 'customer_group', 'customer', 'sales_channel_id' ]],
			
			'customers_groups' => ['primary_key' => ['id'], 'foreign_key' => ['groupid', 'customer_id'], 'operator_str' => [0 => db_prefix().'sales_channel_detailt.customer_group = '.db_prefix().'customers_groups.id' ]],
			'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'sales_channel_detailt.customer = '.db_prefix().'clients.userid'],
			'sales_channel' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'sales_channel_detailt.sales_channel_id = '.db_prefix().'sales_channel.id'],

		]];


		// woocommere_store
		$tables[]=['name' => 'woocommere_store', 'label' => _l('tblwoocommere_store'), 'value' => [
			'woocommere_store' => ['primary_key' => ['id'], 'foreign_key' => [ ]],
			
		]];

		// woocommere_store_detailt
		$tables[]=['name' => 'woocommere_store_detailt', 'label' => _l('tblwoocommere_store_detailt'), 'value' => [
			'woocommere_store_detailt' => ['primary_key' => ['id'], 'foreign_key' => ['group_product_id', 'product_id', 'woocommere_store_id' ]],
			
			'items_groups' => ['primary_key' => ['country_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'woocommere_store_detailt.group_product_id = '.db_prefix().'items_groups.id'],
			'items' => ['primary_key' => ['id'], 'foreign_key' => ['tax', 'tax2', 'group_id', 'color_id', 'style_id', 'model_id', 'size_id', 'unit_id', 'sub_group'], 'operator_str' => db_prefix().'woocommere_store_detailt.product_id = '.db_prefix().'items.id'],
			'woocommere_store' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'woocommere_store_detailt.woocommere_store_id = '.db_prefix().'woocommere_store.id'],

		]];

		// cart
		$tables[]=['name' => 'cart', 'label' => _l('tblcart'), 'value' => [
			'cart' => ['primary_key' => ['id'], 'foreign_key' => ['channel_id', 'userid', 'number_invoice', 'seller', 'allowed_payment_modes', 'warehouse_id', 'shipping_country' ]],
			
			'sales_channel' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'cart.channel_id = '.db_prefix().'sales_channel.id'],
			'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'cart.userid = '.db_prefix().'clients.userid'],
			'invoices' => ['primary_key' => ['id'], 'foreign_key' => ['clientid', 'currency', 'sale_agent', 'project_id'], 'operator_str' => db_prefix().'cart.number_invoice = '.db_prefix().'invoices.id'],
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'cart.seller = '.db_prefix().'staff.staffid' ]],
			'warehouse' => ['primary_key' => ['warehouse_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'cart.warehouse_id = '.db_prefix().'warehouse.warehouse_id'],
			'countries' => ['primary_key' => ['country_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'cart.shipping_country = '.db_prefix().'countries.country_id'],
			

		]];

		// cart_detailt
		$tables[]=['name' => 'cart_detailt', 'label' => _l('tblcart_detailt'), 'value' => [
			'cart_detailt' => ['primary_key' => ['id'], 'foreign_key' => ['product_id', 'cart_id' ]],
			
			'items' => ['primary_key' => ['id'], 'foreign_key' => ['tax', 'tax2', 'group_id', 'color_id', 'style_id', 'model_id', 'size_id', 'unit_id', 'sub_group'], 'operator_str' => db_prefix().'cart_detailt.product_id = '.db_prefix().'items.id'],

			'cart' => ['primary_key' => ['id'], 'foreign_key' => ['channel_id', 'userid', 'number_invoice', 'seller', 'allowed_payment_modes', 'warehouse_id', 'shipping_country'], 'operator_str' => db_prefix().'cart_detailt.cart_id = '.db_prefix().'cart.id'],

		]];
		
		// omni_trade_discount
		$tables[]=['name' => 'omni_trade_discount', 'label' => _l('tblomni_trade_discount'), 'value' => [
			'omni_trade_discount' => ['primary_key' => ['id'], 'foreign_key' => ['group_clients', 'clients', 'group_items', 'items', 'channel' ]],
			
			'customers_groups' => ['primary_key' => ['id'], 'foreign_key' => ['groupid', 'customer_id'], 'operator_str' => [0 => db_prefix().'omni_trade_discount.group_clients = '.db_prefix().'customers_groups.id' ]],
			'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'omni_trade_discount.clients = '.db_prefix().'clients.userid'],
			'items_groups' => ['primary_key' => ['country_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'omni_trade_discount.group_items = '.db_prefix().'items_groups.id'],
			'items' => ['primary_key' => ['id'], 'foreign_key' => ['tax', 'tax2', 'group_id', 'color_id', 'style_id', 'model_id', 'size_id', 'unit_id', 'sub_group'], 'operator_str' => db_prefix().'omni_trade_discount.items = '.db_prefix().'items.id'],
			'sales_channel' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'omni_trade_discount.channel = '.db_prefix().'sales_channel.id'],

		]];

		// omni_log_sync_woo
		$tables[]=['name' => 'omni_log_sync_woo', 'label' => _l('tblomni_log_sync_woo'), 'value' => [
			'omni_log_sync_woo' => ['primary_key' => ['id'], 'foreign_key' => ['order_id' ]],
			
			'cart' => ['primary_key' => ['id'], 'foreign_key' => ['channel_id', 'userid', 'number_invoice', 'seller', 'allowed_payment_modes', 'warehouse_id', 'shipping_country'], 'operator_str' => db_prefix().'omni_log_sync_woo.order_id = '.db_prefix().'cart.id'],
			
		]];

		// omni_shift
		$tables[]=['name' => 'omni_shift', 'label' => _l('tblomni_shift'), 'value' => [
			'omni_shift' => ['primary_key' => ['id'], 'foreign_key' => ['staff_id' ]],
			
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'omni_shift.staff_id = '.db_prefix().'staff.staffid' ]],

		]];

		// omni_shift_history
		$tables[]=['name' => 'omni_shift_history', 'label' => _l('tblomni_shift_history'), 'value' => [
			'omni_shift_history' => ['primary_key' => ['id'], 'foreign_key' => ['shift_id','staff_id','customer_id' ]],
			
			'omni_shift' => ['primary_key' => ['id'], 'foreign_key' => ['staff_id' ], 'operator_str' => [0 => db_prefix().'omni_shift_history.shift_id = '.db_prefix().'omni_shift.id' ]],
			'staff' => ['primary_key' => ['staffid'], 'foreign_key' => ['role'], 'operator_str' => [0 => db_prefix().'omni_shift_history.staff_id = '.db_prefix().'staff.staffid' ]],
			'clients' => ['primary_key' => ['userid'], 'foreign_key' => ['country', 'leadid', 'addedfrom', 'default_currency'], 'operator_str' => db_prefix().'omni_shift_history.customer_id = '.db_prefix().'clients.userid'],
			
		]];

		
		// omni_cart_payment
		$tables[]=['name' => 'omni_cart_payment', 'label' => _l('tblomni_cart_payment'), 'value' => [
			'omni_cart_payment' => ['primary_key' => ['id'], 'foreign_key' => ['cart_id','payment_id' ]],
			
			'cart' => ['primary_key' => ['id'], 'foreign_key' => ['channel_id', 'userid', 'number_invoice', 'seller', 'allowed_payment_modes', 'warehouse_id', 'shipping_country'], 'operator_str' => db_prefix().'omni_cart_payment.cart_id = '.db_prefix().'cart.id'],
			'payment_modes' => ['primary_key' => ['id'], 'foreign_key' => ['staff_id' ], 'operator_str' => [0 => db_prefix().'omni_cart_payment.payment_id = '.db_prefix().'payment_modes.id' ]],
			
		]];

		// omni_pre_order_product_setting
		$tables[]=['name' => 'omni_pre_order_product_setting', 'label' => _l('tblomni_pre_order_product_setting'), 'value' => [
			'omni_pre_order_product_setting' => ['primary_key' => ['id'], 'foreign_key' => ['channel_id','customer_group','customer','group_product_id' ]],
			
			'sales_channel' => ['primary_key' => ['id'], 'foreign_key' => [], 'operator_str' => db_prefix().'omni_pre_order_product_setting.channel_id = '.db_prefix().'sales_channel.id'],
			'items_groups' => ['primary_key' => ['country_id'], 'foreign_key' => [], 'operator_str' => db_prefix().'omni_pre_order_product_setting.group_product_id = '.db_prefix().'items_groups.id'],

		]];



	}

	//HR records
	if(rb_get_status_modules('hr_profile')){

	}


	// hr payroll
	if(rb_get_status_modules('hr_payroll')){

	}

	// Timesheet
	if(rb_get_status_modules('timesheets')){

	}


	// Recruitment
	if(rb_get_status_modules('recruitment')){

	}


	//Sales commission
	if(rb_get_status_modules('commission')){

	}


	//custome loyaty
	if(rb_get_status_modules('loyalty')){

	}


	//Accountting
	if(rb_get_status_modules('accounting')){

	}
	
	
	
	
	
	
	
	
	
	




	return $tables;
}

/**
 * rb filter type
 * @return [type] 
 */
function rb_filter_type()
{
	$filters=[];
	$filters[]=['name' => 'equal', 'label' => _l('equal'), 'symbol' => '='];
	$filters[]=['name' => 'greater_than', 'label' => _l('greater_than'), 'symbol' => '>'];
	$filters[]=['name' => 'less_than', 'label' => _l('less_than'), 'symbol' => '<'];
	$filters[]=['name' => 'greater_than_or_equal', 'label' => _l('greater_than_or_equal'), 'symbol' => '>='];
	$filters[]=['name' => 'less_than_or_equal', 'label' => _l('less_than_or_equal'), 'symbol' => '<='];
	$filters[]=['name' => 'between', 'label' => _l('between'), 'symbol' => 'and'];
	$filters[]=['name' => 'like', 'label' => _l('like'), 'symbol' => 'like'];
	$filters[]=['name' => 'NOT_like', 'label' => _l('NOT_like'), 'symbol' => 'NOT LIKE'];
	$filters[]=['name' => 'not_equal', 'label' => _l('not_equal'), 'symbol' => '!='];
	$filters[]=['name' => 'begin_with', 'label' => _l('begin_with'), 'symbol' => 'LIKE _%'];
	$filters[]=['name' => 'end_with', 'label' => _l('end_with'), 'symbol' => 'LIKE %_'];
	$filters[]=['name' => 'in', 'label' => _l('in'), 'symbol' => 'IN ()'];
	$filters[]=['name' => 'not_in', 'label' => _l('not_in'), 'symbol' => 'NOT IN ()'];

	return $filters;
}

/**
 * rb cell type
 * @return [type] 
 */
function rb_cell_type()
{
	$cell_types=[];
	$cell_types[] = [
		'id' => 'standart_cell',
		'label' => _l('rb_standart_cell'),
	];
	
	$cell_types[] = [
		'id' => 'image_cell',
		'label' => _l('rb_image_cell'),
	];

	$cell_types[] = [
		'id' => 'append_a_text',
		'label' => _l('rb_append_a_text'),
	];

	$cell_types[] = [
		'id' => 'prepend_a_text',
		'label' => _l('rb_prepend_a_text'),
	];
	

	return $cell_types;
}

/**
 * Gets the fields by template identifier.
 *
 * @param        $template  The template
 */
function get_fields_by_template_id($template, $select = false){
	$CI = &get_instance();
	$CI->db->select('*,CONCAT(field_name,"-",table_name) as label');
	$CI->db->where('templates_id', $template);
	$rb_columns =  $CI->db->get(db_prefix().'rb_columns')->result_array();

	if($select){

		foreach ($rb_columns as $key => $list_field) {

			$rb_columns[$key]['field_name'] = _l('tbl'.$list_field['table_name'].'_'.$list_field['field_name']).' ('._l('tbl'.$list_field['table_name']).')';
		}
	}

	return $rb_columns;
}

/**
 * rb subtotals data
 * @return [type] 
 */
function rb_subtotals_data()
{
	$subtotal_data=[];
	$subtotal_data[] = [
		'name' => 'sum',
		'label' => _l('rb_sum'),
	];
	
	$subtotal_data[] = [
		'name' => 'count',
		'label' => _l('rb_count'),
	];

	$subtotal_data[] = [
		'name' => 'average',
		'label' => _l('rb_average'),
	];

	$subtotal_data[] = [
		'name' => 'min',
		'label' => _l('rb_min'),
	];

	$subtotal_data[] = [
		'name' => 'max',
		'label' => _l('rb_max'),
	];
	

	return $subtotal_data;
}

/**
 * rb join type
 * @return [type] 
 */
function rb_join_type()
{
	$join_type_data=[];
	
	$join_type_data[] = [
		'name' => 'left_join',
		'label' => _l('rb_left_join'),
	];

	$join_type_data[] = [
		'name' => 'right_join',
		'label' => _l('rb_right_join'),
	];

	return $join_type_data;
}

/**
 * rb check report filter
 * @param  [type] $template_id 
 * @return [type]              
 */
function rb_check_report_filter($template_id)
{
	$CI = &get_instance();

	$ask_user = false;

	$CI->db->select('*');
	$CI->db->where('templates_id', $template_id);
	$data_source_filters = $CI->db->get(db_prefix().'rb_data_source_filters')->result_array();

	foreach ($data_source_filters as $filter) {
	    if($filter['ask_user'] == 'yes'){
	    	$ask_user = true;

	    	return $ask_user;
	    }
	}

	return $ask_user;
}

/**
 * rb field name date input
 * @return [type] 
 */
function rb_field_name_date_input()
{
	$name_date_input=[];
	$name_date_input[] = 'datecreated';

	return $name_date_input;
}

/**
 * rb get status modules
 * @param  [type] $module_name 
 * @return [type]              
 */
function rb_get_status_modules($module_name){
	$CI             = &get_instance();

	$sql = 'select * from '.db_prefix().'modules where module_name = "'.$module_name.'" AND active =1 ';
	$module = $CI->db->query($sql)->row();
	if($module){
		return true;
	}else{
		return false;
	}
}

/**
 * rb report get children data
 * @param  [type] $id            
 * @param  [type] $search        
 * @param  array  $group_by_data 
 * @param  array  $where         
 * @return [type]                
 */
function rb_report_get_children_data($id, $search, $group_by_data = [], $where = [])
{
	$CI             = &get_instance();
	$rb_primary_foreign_key_field = rb_primary_foreign_key_field();

	$where_data = [];

	foreach ($group_by_data as $column_name) {
		if(isset($where[$column_name])){

			if(isset($rb_primary_foreign_key_field[ new_str_replace('.', '_', $column_name)])){
        		//number
        		if(!is_numeric($where[$column_name])){
        			$condition_value = "'".$where[$column_name]."'";
        		}else{
        			$condition_value = $where[$column_name];
        		}
			}else{
        		//text
				$condition_value = "'".$where[$column_name]."'";
			}

			$where_data[] = 'AND'.' '.$column_name.' = '.$condition_value;

		}
	}
	$children_report_date = $CI->report_builder_model->create_children_report_from_template($id, $search, $where_data);

	return $children_report_date['rResult'];
}

/**
 * rb get role name
 * @param  [type] $role_id 
 * @return [type]          
 */
function rb_get_role_name($role_id)
{
	$CI             = &get_instance();
	$CI->db->where('roleid', $role_id);
	$CI->db->select('name');
	$role = $CI->db->get(db_prefix().'roles')->row();

	if($role){
		return $role->name; 
	}else{
		return ''; 
	}

}

/**
 * rb get department name
 * @param  [type] $department_id 
 * @return [type]                
 */
function rb_get_department_name($department_id)
{
	$CI             = &get_instance();
	$CI->db->where('departmentid', $department_id);
	$CI->db->select('name');
	$department = $CI->db->get(db_prefix().'departments')->row();

	if($department){
		return $department->name; 
	}else{
		return ''; 
	}

}

/**
 * rb primary foreign key field
 * @return [type] 
 */
function rb_primary_foreign_key_field()
{
	// operator: = ; IN ; NOT IN : 
	//input type: select single, muiltiple
	$fields = [];


	//tblclients
	$fields['tblclients_userid']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblclients_country']=['table_name' => 'countries', 'field_name' => 'country_id', 'label' => 'short_name'];
	$fields['tblclients_active']=['table_name' => 'clients', 'field_name' => [0 => ['value' => 0, 'label' => _l('unactive')], 1 => ['value' => 1, 'label' => _l('active')], ], 'label' => 'name']; //TODO
	$fields['tblclients_leadid']=['table_name' => 'leads', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblclients_default_currency']=['table_name' => 'currencies', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblclients_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tbl countries
	$fields['tblcountries_country_id']=['table_name' => 'countries', 'field_name' => 'country_id', 'label' => 'short_name'];

	//tblcontact
	$fields['tblcontacts_id']=['table_name' => 'contacts', 'field_name' => 'id', 'label' => 'firstname#lastname'];
	$fields['tblcontacts_userid']=['table_name' => 'contacts', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblcontacts_active']=['table_name' => 'contacts', 'field_name' => [0 => ['value' => 0, 'label' => _l('unactive')], 1 => ['value' => 1, 'label' => _l('active')], ], 'label' => 'name']; //TODO
	
	//tblcustomer_groups
	$fields['tblcustomers_groups_id']=['table_name' => 'customers_groups', 'field_name' => 'id', 'label' => 'name'];

	//tblinvoices
	$fields['tblinvoices_id']=['table_name' => 'invoices', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblinvoices_clientid']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblinvoices_currency']=['table_name' => 'currencies', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblinvoices_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblinvoices_sale_agent']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblinvoices_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblinvoices_subscription_id']=['table_name' => 'subscriptions', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblinvoices_sent']=['table_name' => 'invoices', 'field_name' => [0 => ['value' => 0, 'label' => _l('notsend')], 1 => ['value' => 1, 'label' => _l('sent')], ], 'label' => 'name']; //TODO
	$fields['tblinvoices_status'] = ['table_name' => 'invoices', 'field_name' => [0 => ['value' => 1, 'label' => _l('invoice_status_unpaid')], 1 => ['value' => 2, 'label' => _l('invoice_status_paid')], 2 => ['value' => 3, 'label' => _l('invoice_status_not_paid_completely')], 3 => ['value' => 4, 'label' => _l('invoice_status_overdue')], 4 => ['value' => 5, 'label' => _l('invoice_status_cancelled')], 5 => ['value' => 6, 'label' => _l('invoice_status_draft')] ], 'label' => 'name'];

	//tblestimates
	$fields['tblestimates_id']=['table_name' => 'estimates', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblestimates_sent']=['table_name' => 'estimates', 'field_name' => [0 => ['value' => 0, 'label' => _l('notsend')], 1 => ['value' => 1, 'label' => _l('sent')], ], 'label' => 'name']; //TODO
	$fields['tblestimates_clientid']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblestimates_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblestimates_currency']=['table_name' => 'currencies', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblestimates_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblestimates_status']=['table_name' => 'estimates', 'field_name' => [0 => ['value' => 1, 'label' => _l('STATUS_DRAFT')], 1 => ['value' => 2, 'label' => _l('STATUS_SENT')], 2 => ['value' => 3, 'label' => _l('STATUS_DECLINED')], 3 => ['value' => 4, 'label' => _l('STATUS_ACCEPTED')], 4 => ['value' => 5, 'label' => _l('STATUS_EXPIRED')]  ], 'label' => 'name'];
	$fields['tblestimates_invoiceid']=['table_name' => 'invoices', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblestimates_sale_agent']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	
	//tblcreditnotes
	$fields['tblcreditnotes_id']=['table_name' => 'creditnotes', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblcreditnotes_clientid']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblcreditnotes_currency']=['table_name' => 'currencies', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblcreditnotes_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblcreditnotes_status']=['table_name' => 'creditnotes', 'field_name' => [0 => ['value' => 1, 'label' => _l('STATUS_OPEN')], 1 => ['value' => 2, 'label' => _l('STATUS_CLOSED')], 2 => ['value' => 3, 'label' => _l('STATUS_VOID')] ], 'label' => 'name'];
	$fields['tblestimates_invoiceid']=['table_name' => 'invoices', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblcreditnotes_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];


	//tbltickets
	$fields['tbltickets_ticketid']=['table_name' => 'tickets', 'field_name' => 'ticketid', 'label' => 'ticketid'];
	$fields['tbltickets_adminreplying']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tbltickets_userid']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tbltickets_contactid']=['table_name' => 'contacts', 'field_name' => 'id', 'label' => 'firstname#lastname'];
	$fields['tbltickets_department']=['table_name' => 'departments', 'field_name' => 'departmentid', 'label' => 'name'];
	$fields['tbltickets_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tbltickets_assigned']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tbltickets_status'] = ['table_name' => 'tickets', 'field_name' => [0 => ['value' => 1, 'label' => _l('ticket_status_db_1')], 1 => ['value' => 2, 'label' => _l('ticket_status_db_2')], 2 => ['value' => 3, 'label' => _l('ticket_status_db_3')] , 3 => ['value' => 4, 'label' => _l('ticket_status_db_4')] , 4 => ['value' => 5, 'label' => _l('ticket_status_db_5')] ], 'label' => 'name'];


	//subscriptions
	$fields['tblsubscriptions_id']=['table_name' => 'subscriptions', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblsubscriptions_clientid']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblsubscriptions_currency']=['table_name' => 'currencies', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblsubscriptions_tax_id']=['table_name' => 'taxes', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblsubscriptions_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblsubscriptions_created_from']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblsubscriptions_status']=['table_name' => 'subscriptions', 'field_name' => [0 => ['value' => 'subscription_status_active', 'label' => _l('subscription_active')], 1 => ['value' => 'subscription_status_future', 'label' => _l('subscription_future')], 2 => ['value' => 'subscription_status_past_due', 'label' => _l('subscription_past_due')] , 3 => ['value' => 'subscription_status_unpaid', 'label' => _l('subscription_unpaid')] , 4 => ['value' => 'subscription_status_incomplete', 'label' => _l('subscription_incomplete')] , 5 => ['value' => 'subscription_status_canceled', 'label' => _l('invoice_status_cancelled')] , 6 => ['value' => 'subscription_status_incomplete_expired', 'label' => _l('subscription_incomplete_expired')] , 7 => ['value' => 'not_subscribed', 'label' => _l('subscription_not_subscribed')] ], 'label' => 'name'];


	//tblcontracts
	$fields['tblcontracts_id']=['table_name' => 'contracts', 'field_name' => 'id', 'label' => 'subject'];
	$fields['tblcontracts_client']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblcontracts_contract_type']=['table_name' => 'contracts_types', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblcontracts_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblcontracts_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];


	//tbltasks
	$fields['tbltasks_id']=['table_name' => 'tasks', 'field_name' => 'id', 'label' => 'name'];
	$fields['tbltasks_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tbltasks_invoice_id']=['table_name' => 'invoices', 'field_name' => 'id', 'label' => 'id'];
	$fields['tbltasks_status']=['table_name' => 'tasks', 'field_name' => [0 => ['value' => 1, 'label' => _l('task_status_1')], 1 => ['value' => 2, 'label' => _l('task_status_2')], 2 => ['value' => 3, 'label' => _l('task_status_3')] , 3 => ['value' => 4, 'label' => _l('task_status_4')] , 4 => ['value' => 5, 'label' => _l('task_status_5')] ], 'label' => 'name'];

	//tbltask_comments
	$fields['tbltask_comments_id']=['table_name' => 'task_comments', 'field_name' => 'id', 'label' => 'content'];
	$fields['tbltask_comments_taskid']=['table_name' => 'tasks', 'field_name' => 'id', 'label' => 'name'];
	$fields['tbltask_comments_staffid']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tbltask_comments_contact_id']=['table_name' => 'contacts', 'field_name' => 'id', 'label' => 'firstname#lastname'];
	$fields['tbltask_comments_file_id']=['table_name' => 'files', 'field_name' => 'id', 'label' => 'file_name'];

	//tblreminders
	$fields['tblreminders_id']=['table_name' => 'reminders', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblreminders_staff']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblreminders_creator']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblproposals
	$fields['tblproposals_id']=['table_name' => 'proposals', 'field_name' => 'id', 'label' => 'subject'];
	$fields['tblproposals_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblproposals_assigned']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblproposals_country']=['table_name' => 'countries', 'field_name' => 'country_id', 'label' => 'short_name'];
	$fields['tblproposals_status']=['table_name' => 'proposals', 'field_name' => [0 => ['value' => 1, 'label' => _l('proposal_status_open')], 1 => ['value' => 2, 'label' => _l('proposal_status_declined')], 2 => ['value' => 3, 'label' => _l('proposal_status_accepted')] , 3 => ['value' => 4, 'label' => _l('proposal_status_sent')] , 4 => ['value' => 5, 'label' => _l('proposal_status_revised')], 5 => ['value' => 6, 'label' => _l('estimate_status_draft')] ], 'label' => 'name'];
	$fields['tblproposals_estimate_id']=['table_name' => 'estimates', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblproposals_invoice_id']=['table_name' => 'invoices', 'field_name' => 'id', 'label' => 'id'];

	//tblexpenses
	$fields['tblexpenses_id']=['table_name' => 'expenses', 'field_name' => 'id', 'label' => 'expense_name'];
	$fields['tblexpenses_currency']=['table_name' => 'currencies', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblexpenses_clientid']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblexpenses_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblexpenses_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblleads
	$fields['tblleads_id']=['table_name' => 'leads', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblleads_country']=['table_name' => 'countries', 'field_name' => 'country_id', 'label' => 'short_name'];
	$fields['tblleads_assigned']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblleads_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblleads_client_id']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];

	//tblprojects
	$fields['tblprojects_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblprojects_clientid']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblprojects_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblprojects_status']=['table_name' => 'projects', 'field_name' => [0 => ['value' => 1, 'label' => _l('project_status_1')], 1 => ['value' => 2, 'label' => _l('project_status_2')], 2 => ['value' => 3, 'label' => _l('project_status_3')] , 3 => ['value' => 4, 'label' => _l('project_status_4')] , 4 => ['value' => 5, 'label' => _l('project_status_5')] ], 'label' => 'name'];

	//tblprojectdiscussions
	$fields['tblprojectdiscussions_id']=['table_name' => 'projectdiscussions', 'field_name' => 'id', 'label' => 'subject'];
	$fields['tblprojectdiscussions_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblprojectdiscussions_contact_id']=['table_name' => 'contacts', 'field_name' => 'id', 'label' => 'firstname#lastname'];
	$fields['tblprojectdiscussions_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblprojectdiscussioncomments
	$fields['tblprojectdiscussioncomments_id']=['table_name' => 'projectdiscussioncomments', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblprojectdiscussioncomments_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblprojectdiscussioncomments_contact_id']=['table_name' => 'contacts', 'field_name' => 'id', 'label' => 'firstname#lastname'];
	$fields['tblprojectdiscussioncomments_discussion_id']=['table_name' => 'projectdiscussions', 'field_name' => 'id', 'label' => 'subject'];

	//tblstaff
	$fields['tblstaff_staffid']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblstaff_role']=['table_name' => 'roles', 'field_name' => 'roleid', 'label' => 'name'];
	$fields['tblstaff_team_manage']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblstaff_status_work']=['table_name' => 'staff', 'field_name' => [0 => ['value' => 'working', 'label' => _l('working')], 1 => ['value' => 'maternity_leave', 'label' => _l('maternity_leave')], 2 => ['value' => 'inactivity', 'label' => _l('inactivity')] ], 'label' => 'name'];

	//tblproposal_comments
	$fields['tblproposal_comments_id']=['table_name' => 'proposal_comments', 'field_name' => 'id', 'label' => 'content'];
	$fields['tblproposal_comments_proposalid']=['table_name' => 'proposals', 'field_name' => 'id', 'label' => 'subject'];
	$fields['tblproposal_comments_staffid']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblitem_tax
	$fields['tblitem_tax_id']=['table_name' => 'item_tax', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblitem_tax_itemid']=['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];

	//tblitemable
	$fields['tblitemable_id']=['table_name' => 'itemable', 'field_name' => 'id', 'label' => 'id'];

	//tblcurrencies
	$fields['tblcurrencies_id']=['table_name' => 'currencies', 'field_name' => 'id', 'label' => 'name'];

	//tblnotes
	$fields['tblnotes_id']=['table_name' => 'notes', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblnotes_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblinvoicepaymentrecords
	$fields['tblinvoicepaymentrecords_id']=['table_name' => 'invoicepaymentrecords', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblinvoicepaymentrecords_invoiceid']=['table_name' => 'invoices', 'field_name' => 'id', 'label' => 'id'];

	//tblcredits
	$fields['tblcredits_id']=['table_name' => 'credits', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblcredits_invoice_id']=['table_name' => 'invoices', 'field_name' => 'id', 'label' => 'id'];
	// $fields['tblcredits_credit_id']=['table_name' => 'credits', 'field_name' => 'credit_id'];
	$fields['tblcredits_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblrelated_items
	$fields['tblrelated_items_id']=['table_name' => 'related_items', 'field_name' => 'id'];
	$fields['tblrelated_items_item_id']=['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];

	//tblpayment_modes
	$fields['tblpayment_modes_id']=['table_name' => 'payment_modes', 'field_name' => 'id', 'label' => 'name'];

	//tblcreditnote_refunds
	$fields['tblcreditnote_refunds_id']=['table_name' => 'creditnote_refunds', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblcreditnote_refunds_credit_note_id']=['table_name' => 'creditnotes', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblcreditnote_refunds_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblitems
	$fields['tblitems_id']=['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblitems_tax']=['table_name' => 'taxes', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblitems_tax2']=['table_name' => 'taxes', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblitems_group_id']=['table_name' => 'items_groups', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblitems_color_id']=['table_name' => 'ware_color', 'field_name' => 'color_id', 'label' => 'color_name'];
	$fields['tblitems_style_id']=['table_name' => 'ware_style_type', 'field_name' => 'style_type_id', 'label' => 'style_name'];
	$fields['tblitems_model_id']=['table_name' => 'ware_body_type', 'field_name' => 'body_type_id', 'label' => 'body_name'];
	$fields['tblitems_size_id']=['table_name' => 'ware_size_type', 'field_name' => 'size_type_id', 'label' => 'size_name'];
	$fields['tblitems_unit_id']=['table_name' => 'ware_unit_type', 'field_name' => 'unit_type_id', 'label' => 'unit_name'];
	$fields['tblitems_sub_group']=['table_name' => 'wh_sub_group', 'field_name' => 'id', 'label' => 'sub_group_name'];

	//tblitems_groups
	$fields['tblitems_groups_id']=['table_name' => 'items_groups', 'field_name' => 'id', 'label' => 'name'];

	//tbltaxes
	$fields['tbltaxes_id']=['table_name' => 'taxes', 'field_name' => 'id', 'label' => 'name'];

	//tblexpenses_categories
	$fields['tblexpenses_categories_id']=['table_name' => 'expenses_categories', 'field_name' => 'id', 'label' => 'name'];
	
	//tblcontracts_types
	$fields['tblcontracts_types_id']=['table_name' => 'contracts_types', 'field_name' => 'id', 'label' => 'name'];

	//tblcontract_comments
	$fields['tblcontract_comments_id']=['table_name' => 'contract_comments', 'field_name' => 'id'];
	$fields['tblcontract_comments_contract_id']=['table_name' => 'contracts', 'field_name' => 'id', 'label' => 'subject'];
	$fields['tblcontract_comments_staffid']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblcontract_renewals
	$fields['tblcontract_renewals_id']=['table_name' => 'contract_renewals', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblcontract_renewals_contractid']=['table_name' => 'contracts', 'field_name' => 'id', 'label' => 'subject'];
	$fields['tblcontract_renewals_renewed_by_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tbltaskstimers
	$fields['tbltaskstimers_id']=['table_name' => 'taskstimers', 'field_name' => 'id', 'label' => 'id'];
	$fields['tbltaskstimers_task_id']=['table_name' => 'tasks', 'field_name' => 'id', 'label' => 'name'];
	$fields['tbltaskstimers_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblproject_members
	$fields['tblproject_members_id']=['table_name' => 'project_members', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblproject_members_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblproject_members_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblpinned_projects
	$fields['tblpinned_projects_id']=['table_name' => 'pinned_projects', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpinned_projects_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblpinned_projects_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblproject_settings
	$fields['tblproject_settings_id']=['table_name' => 'project_settings', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblproject_settings_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];

	//tblmilestones
	$fields['tblmilestones_id']=['table_name' => 'milestones', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblmilestones_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];

	//tbltask_assigned
	$fields['tbltask_assigned_id']=['table_name' => 'task_assigned', 'field_name' => 'id', 'label' => 'id'];
	$fields['tbltask_assigned_staffid']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tbltask_assigned_taskid']=['table_name' => 'tasks', 'field_name' => 'id', 'label' => 'name'];
	$fields['tbltask_assigned_assigned_from']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tbltask_followers
	$fields['tbltask_followers_id']=['table_name' => 'task_followers', 'field_name' => 'id', 'label' => 'id'];
	$fields['tbltask_followers_staffid']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tbltask_followers_taskid']=['table_name' => 'tasks', 'field_name' => 'id', 'label' => 'name'];

	//tblproject_notes
	$fields['tblproject_notes_id']=['table_name' => 'project_notes', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblproject_notes_project_id']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblproject_notes_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];


	/*-----------warehouse module------------*/

	// tblgoods_receipt
	$fields['tblgoods_receipt_id'] = ['table_name' => 'goods_receipt', 'field_name' => 'id', 'label' => 'goods_receipt_code'];
	$fields['tblgoods_receipt_buyer_id'] = ['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblgoods_receipt_pr_order_id'] = ['table_name' => 'pur_orders', 'field_name' => 'id', 'label' => 'pur_order_name'];
	$fields['tblgoods_receipt_addedfrom'] = ['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tbltblgoods_receipt_project'] = ['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblgoods_receipt_approval'] = ['table_name' => 'goods_receipt', 'field_name' => [0 => ['value' => 1, 'label' => _l('not_yet_approval')], 1 => ['value' => 2, 'label' => _l('approval')], 2 => ['value' => 3, 'label' => _l('reject')], ], 'label' => 'name'];
	$fields['tblgoods_receipt_department'] = ['table_name' => 'departments', 'field_name' => 'departmentid', 'label' => 'name'];
	$fields['tblgoods_receipt_requester'] = ['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	// tblgoods_receipt_detail
	$fields['tblgoods_receipt_detail_id'] = ['table_name' => 'goods_receipt_detail', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblgoods_receipt_detail_goods_receipt_id'] = ['table_name' => 'goods_receipt', 'field_name' => 'id', 'label' => 'goods_receipt_code'];
	$fields['tblgoods_receipt_detail_commodity_code'] = ['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblgoods_receipt_detail_commodity_name'] = ['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblgoods_receipt_detail_warehouse_id'] = ['table_name' => 'warehouse', 'field_name' => 'id', 'label' => 'warehouse_name'];
	$fields['tblgoods_receipt_detail_unit_id'] = ['table_name' => 'ware_unit_type', 'field_name' => 'unit_type_id', 'label' => 'unit_name'];
	$fields['tblgoods_receipt_type'] = ['table_name' => 'goods_receipt', 'field_name' => [0 => ['value' => 'capex', 'label' => _l('capex')], 1 => ['value' => 'opex', 'label' => _l('opex')] ], 'label' => 'name'];
	//don't get
	// $fields['tblgoods_receipt_detail_tax'] = ['table_name' => 'taxes', 'field_name' => 'id', 'label' => 'name'];
	// $fields['tblgoods_receipt_warehouse_id'] = ['table_name' => 'taxes', 'field_name' => 'id', 'label' => 'name'];
	
	// tblwh_loss_adjustment
	$fields['tblwh_loss_adjustment_id']=['table_name' => 'wh_loss_adjustment', 'field_name' => 'id', 'label' => 'id#type'];
	$fields['tblwh_loss_adjustment_addfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblwh_loss_adjustment_warehouses']=['table_name' => 'warehouse', 'field_name' => 'warehouse_id', 'label' => 'warehouse_name'];
	$fields['tblwh_loss_adjustment_status']=['table_name' => 'wh_loss_adjustment', 'field_name' => [0 => ['value' => 1, 'label' => _l('not_yet_approval')], 1 => ['value' => 2, 'label' => _l('approval')], 2 => ['value' => 3, 'label' => _l('reject')], ], 'label' => 'name'];

	// tblwh_loss_adjustment_detail
	$fields['tblwh_loss_adjustment_detail_id']=['table_name' => 'wh_loss_adjustment_detail', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblwh_loss_adjustment_detail_items']=['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblwh_loss_adjustment_detail_unit']=['table_name' => 'ware_unit_type', 'field_name' => 'unit_type_id', 'label' => 'unit_name'];
	$fields['tblwh_loss_adjustment_detail_loss_adjustment']=['table_name' => 'wh_loss_adjustment', 'field_name' => 'id', 'label' => 'id#type'];

	// tblgoods_delivery
	$fields['tblgoods_delivery_id']=['table_name' => 'goods_delivery', 'field_name' => 'id', 'label' => 'goods_delivery_code'];
	$fields['tblgoods_delivery_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblgoods_delivery_invoice_id']=['table_name' => 'invoices', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblgoods_delivery_pr_order_id']=['table_name' => 'pur_orders', 'field_name' => 'id', 'label' => 'pur_order_name'];
	$fields['tblgoods_delivery_approval']=['table_name' => 'goods_delivery', 'field_name' => [0 => ['value' => 1, 'label' => _l('not_yet_approval')], 1 => ['value' => 2, 'label' => _l('approval')], 2 => ['value' => 3, 'label' => _l('reject')], ], 'label' => 'name'];
	$fields['tblgoods_delivery_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblgoods_delivery_customer_code']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblgoods_delivery_project']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblgoods_delivery_department']=['table_name' => 'departments', 'field_name' => 'departmentid', 'label' => 'name'];
	$fields['tblgoods_delivery_requester']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblgoods_delivery_detail
	$fields['tblgoods_delivery_detail_id']=['table_name' => 'goods_delivery_detail', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblgoods_delivery_detail_goods_delivery_id']=['table_name' => 'goods_delivery', 'field_name' => 'id', 'label' => 'goods_delivery_code'];
	$fields['tblgoods_delivery_detail_commodity_code']=['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblgoods_delivery_detail_warehouse_id']=['table_name' => 'warehouse', 'field_name' => 'id', 'label' => 'warehouse_name'];
	$fields['tblgoods_delivery_detail_unit_id']=['table_name' => 'ware_unit_type', 'field_name' => 'unit_type_id', 'label' => 'unit_name'];

	//tblinternal_delivery_note
	$fields['tblinternal_delivery_note_id']=['table_name' => 'internal_delivery_note', 'field_name' => 'id', 'label' => 'internal_delivery_code'];
	$fields['tblinternal_delivery_note_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblinternal_delivery_note_approval']=['table_name' => 'internal_delivery_note', 'field_name' => [0 => ['value' => 1, 'label' => _l('not_yet_approval')], 1 => ['value' => 2, 'label' => _l('approval')], 2 => ['value' => 3, 'label' => _l('reject')], ], 'label' => 'name'];
	$fields['tblinternal_delivery_note_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	//tblinternal_delivery_note_detail
	$fields['tblinternal_delivery_note_detail_id']=['table_name' => 'internal_delivery_note_detail', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblinternal_delivery_note_detail_internal_delivery_id']=['table_name' => 'internal_delivery_note', 'field_name' => 'id', 'label' => 'internal_delivery_code'];
	$fields['tblinternal_delivery_note_detail_commodity_code']=['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblinternal_delivery_note_detail_from_stock_name']=['table_name' => 'warehouse', 'field_name' => 'id', 'label' => 'warehouse_name'];
	$fields['tblinternal_delivery_note_detail_to_stock_name']=['table_name' => 'warehouse', 'field_name' => 'id', 'label' => 'warehouse_name'];
	$fields['tblinternal_delivery_note_detail_unit_id']=['table_name' => 'ware_unit_type', 'field_name' => 'unit_type_id', 'label' => 'unit_name'];

	//tblwarehouse
	$fields['tblwarehouse_warehouse_id']=['table_name' => 'warehouse', 'field_name' => 'warehouse_id', 'label' => 'warehouse_code'];
	$fields['tblwarehouse_country']=['table_name' => 'countries', 'field_name' => 'country_id', 'label' => 'short_name'];

	//tblware_commodity_type
	$fields['tblware_commodity_type_commodity_type_id']=['table_name' => 'ware_commodity_type', 'field_name' => 'commodity_type_id', 'label' => 'commondity_code'];
	$fields['tblware_commodity_type_display']=['table_name' => 'ware_commodity_type', 'field_name' => [0 => ['value' => 0, 'label' => _l('not_display')], 1 => ['value' => 1, 'label' => _l('display')], ], 'label' => 'commondity_code'];

	//tblwh_sub_group
	$fields['tblwh_sub_group_id']=['table_name' => 'wh_sub_group', 'field_name' => 'id', 'label' => 'sub_group_code'];
	$fields['tblwh_sub_group_group_id']=['table_name' => 'items_groups', 'field_name' => 'id', 'label' => 'name'];

	//tblware_unit_type
	$fields['tblware_unit_type_unit_type_id']=['table_name' => 'ware_unit_type', 'field_name' => 'unit_type_id', 'label' => 'unit_code'];
	$fields['tblware_unit_type_display']=['table_name' => 'ware_unit_type', 'field_name' => [0 => ['value' => 0, 'label' => _l('not_display')], 1 => ['value' => 1, 'label' => _l('display')], ], 'label' => 'unit_code'];

	//tblware_color
	$fields['tblware_color_color_id']=['table_name' => 'ware_color', 'field_name' => 'color_id', 'label' => 'color_code'];
	$fields['tblware_color_display']=['table_name' => 'ware_color', 'field_name' => [0 => ['value' => 0, 'label' => _l('not_display')], 1 => ['value' => 1, 'label' => _l('display')], ], 'label' => 'color_code'];

	//tblware_body_type
	$fields['tblware_body_type_body_type_id']=['table_name' => 'ware_body_type', 'field_name' => 'body_type_id', 'label' => 'body_code'];
	$fields['tblware_body_type_display']=['table_name' => 'ware_body_type', 'field_name' => [0 => ['value' => 0, 'label' => _l('not_display')], 1 => ['value' => 1, 'label' => _l('display')], ], 'label' => 'body_code'];

	//tblware_size_type
	$fields['tblware_size_type_size_type_id']=['table_name' => 'ware_size_type', 'field_name' => 'size_type_id', 'label' => 'size_code'];
	$fields['tblware_size_type_display']=['table_name' => 'ware_size_type', 'field_name' => [0 => ['value' => 0, 'label' => _l('not_display')], 1 => ['value' => 1, 'label' => _l('display')], ], 'label' => 'size_code'];

	//tblware_style_type
	$fields['tblware_style_type_style_type_id']=['table_name' => 'ware_style_type', 'field_name' => 'style_type_id', 'label' => 'style_code'];
	$fields['tblware_style_type_display']=['table_name' => 'ware_style_type', 'field_name' => [0 => ['value' => 0, 'label' => _l('not_display')], 1 => ['value' => 1, 'label' => _l('display')], ], 'label' => 'style_code'];

	//tblinventory_manage
	$fields['tblinventory_manage_id']=['table_name' => 'inventory_manage', 'field_name' => 'id'];
	$fields['tblinventory_manage_warehouse_id']=['table_name' => 'warehouse', 'field_name' => 'id', 'label' => 'warehouse_name'];
	$fields['tblinventory_manage_commodity_id']=['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];

	// tblpur_comments
	$fields['tblpur_comments_id']=['table_name' => 'pur_comments', 'field_name' => 'id'];
	$fields['tblpur_comments_staffid']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];


	// tblpur_contacts
	$fields['tblpur_contacts_id']=['table_name' => 'pur_contacts', 'field_name' => 'id'];
	$fields['tblpur_contacts_userid']=['table_name' => 'pur_vendor', 'field_name' => 'userid', 'label' => 'firstname#lastname'];
	$fields['tblpur_contacts_active']=['table_name' => 'pur_vendor', 'field_name' => [0 => ['value' => 0, 'label' => _l('unactive')], 1 => ['value' => 1, 'label' => _l('active')], ], 'label' => 'firstname#lastname'];

	//tblpur_contracts
	$fields['tblpur_contracts_id']=['table_name' => 'pur_contracts', 'field_name' => 'id', 'label' => 'contract_name'];
	$fields['tblpur_contracts_vendor']=['table_name' => 'pur_vendor', 'field_name' => 'userid', 'label' => 'firstname#lastname'];
	$fields['tblpur_contracts_pur_order']=['table_name' => 'pur_orders', 'field_name' => 'id', 'label' => 'pur_order_name'];
	$fields['tblpur_contracts_project']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblpur_contracts_department']=['table_name' => 'departments', 'field_name' => 'departmentid', 'label' => 'name'];
	$fields['tblpur_contracts_add_from']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblpur_contracts_signed']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblpur_contracts_signer']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblpur_contracts_buyer']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];


	// tblpur_debit_notes
	$fields['tblpur_debit_notes_id']=['table_name' => 'pur_debit_notes', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_debit_notes_vendorid']=['table_name' => 'pur_vendor', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblpur_debit_notes_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	// tblpur_debits
	$fields['tblpur_debits_id']=['table_name' => 'pur_debits', 'field_name' => 'id'];
	$fields['tblpur_debits_invoice_id']=['table_name' => 'pur_invoices', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_debits_debit_id']=['table_name' => 'pur_debit_notes', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_debits_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	// tblpur_debits_refunds
	$fields['tblpur_debits_refunds_id']=['table_name' => 'pur_debits_refunds', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_debits_refunds_debit_note_id']=['table_name' => 'pur_debit_notes', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_debits_refunds_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];


	// tblpur_estimates
	$fields['tblpur_estimates_id']=['table_name' => 'pur_estimates', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_estimates_vendor']=['table_name' => 'pur_vendor', 'field_name' => 'userid', 'label' => 'firstname#lastname'];
	$fields['tblpur_estimates_pur_request']=['table_name' => 'pur_request', 'field_name' => 'id', 'label' => 'pur_rq_code'];
	$fields['tblpur_estimates_buyer']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblpur_estimates_status']=['table_name' => 'pur_estimates', 'field_name' => [0 => ['value' => 1, 'label' => _l('not_yet_approval')], 1 => ['value' => 2, 'label' => _l('approval')], 2 => ['value' => 3, 'label' => _l('reject')] ], 'label' => 'firstname#lastname'];

	// tblpur_estimate_detail
	$fields['tblpur_estimate_detail_id']=['table_name' => 'pur_estimate_detail', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_estimate_detail_pur_estimate']=['table_name' => 'pur_estimates', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_estimate_detail_item_code']=['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblpur_estimate_detail_unit_id']=['table_name' => 'ware_unit_type', 'field_name' => 'unit_type_id', 'label' => 'unit_code'];


	// tblpur_invoices
	$fields['tblpur_invoices_id']=['table_name' => 'pur_invoices', 'field_name' => 'id', 'label'];
	$fields['tblpur_invoices_vendor']=['table_name' => 'pur_vendor', 'field_name' => 'userid', 'label' => 'firstname#lastname'];
	$fields['tblpur_invoices_add_from']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblpur_invoices_contract']=['table_name' => 'pur_contracts', 'field_name' => 'id', 'label' => 'contract_name'];
	$fields['tblpur_invoices_pur_order']=['table_name' => 'pur_orders', 'field_name' => 'id', 'label' => 'pur_order_name'];


	// tblpur_invoice_payment
	$fields['tblpur_invoice_payment_id']=['table_name' => 'pur_invoice_payment', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_invoice_payment_pur_invoice']=['table_name' => 'pur_invoices', 'field_name' => 'id', 'label'];
	$fields['tblpur_invoice_payment_requester']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	// tblpur_orders
	$fields['tblpur_orders_id']=['table_name' => 'pur_orders', 'field_name' => 'id', 'label' => 'pur_order_name'];
	$fields['tblpur_orders_vendor']=['table_name' => 'pur_vendor', 'field_name' => 'userid', 'label' => 'firstname#lastname'];
	$fields['tblpur_orders_estimate']=['table_name' => 'pur_estimates', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_orders_buyer']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblpur_orders_clients']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblpur_orders_project']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblpur_orders_pur_request']=['table_name' => 'pur_request', 'field_name' => 'id', 'label' => 'pur_rq_code'];
	$fields['tblpur_orders_department']=['table_name' => 'departments', 'field_name' => 'departmentid', 'label' => 'name'];
	$fields['tblpur_orders_sale_invoice']=['table_name' => 'invoices', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_orders_approve_status']=['table_name' => 'pur_orders', 'field_name' => [0 => ['value' => 1, 'label' => _l('not_yet_approval')], 1 => ['value' => 2, 'label' => _l('approval')], 2 => ['value' => 3, 'label' => _l('reject')], ], 'label' => 'name'];
	$fields['tblpur_orders_delivery_status']=['table_name' => 'pur_orders', 'field_name' => [0 => ['value' => 0, 'label' => _l('undelivered')], 1 => ['value' => 2, 'label' => _l('pending_delivered')], 2 => ['value' => 3, 'label' => _l('partially_delivered')], ], 'label' => 'name'];


	// tblpur_order_detail
	$fields['tblpur_order_detail_id']=['table_name' => 'pur_order_detail', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_order_detail_pur_order']=['table_name' => 'pur_orders', 'field_name' => 'id', 'label' => 'pur_order_name'];
	$fields['tblpur_order_detail_item_code']=['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblpur_order_detail_unit_id']=['table_name' => 'ware_unit_type', 'field_name' => 'unit_type_id', 'label' => 'unit_code'];

	// tblpur_request
	$fields['tblpur_request_id']=['table_name' => 'pur_request', 'field_name' => 'id', 'label' => 'pur_rq_code'];
	$fields['tblpur_request_requester']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblpur_request_department']=['table_name' => 'departments', 'field_name' => 'departmentid', 'label' => 'name'];
	$fields['tblpur_request_project']=['table_name' => 'projects', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblpur_request_sale_invoice']=['table_name' => 'invoices', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblpur_request_status']=['table_name' => 'pur_request', 'field_name' => [0 => ['value' => 1, 'label' => _l('not_yet_approval')], 1 => ['value' => 2, 'label' => _l('approval')], 2 => ['value' => 3, 'label' => _l('reject')], ], 'label' => 'pur_rq_code'];


	// tblpur_request_detail
	$fields['tblpur_request_detail_prd_id']=['table_name' => 'pur_request_detail', 'field_name' => 'prd_id', 'label' => 'prd_id'];
	$fields['tblpur_request_detail_pur_request']=['table_name' => 'pur_request', 'field_name' => 'id', 'label' => 'pur_rq_code'];
	$fields['tblpur_request_detail_item_code']=['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblpur_request_detail_unit_id']=['table_name' => 'ware_unit_type', 'field_name' => 'unit_type_id', 'label' => 'unit_code'];

	// tblpur_vendor
	$fields['tblpur_vendor_userid']=['table_name' => 'pur_vendor', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblpur_vendor_addedfrom']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblpur_vendor_category']=['pur_vendor_cate' => 'pur_vendor', 'field_name' => 'id', 'label' => 'category_name'];
	$fields['tblpur_vendor_active']=['table_name' => 'pur_vendor', 'field_name' => [0 => ['value' => 0, 'label' => _l('unactive')], 1 => ['value' => 1, 'label' => _l('active')], ], 'label' => 'company'];

	// tblpur_vendor_admin
	$fields['tblpur_vendor_admin_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblpur_vendor_admin_vendor_id']=['table_name' => 'pur_vendor', 'field_name' => 'userid', 'label' => 'company'];

	// tblpur_vendor_cate
	$fields['tblpur_vendor_cate_id']=['table_name' => 'pur_vendor_cate', 'field_name' => 'id', 'label' => 'category_name'];


	/*--------omnisales module-------*/

	// tblsales_channel
	$fields['tblsales_channel_id']=['table_name' => 'sales_channel', 'field_name' => 'id', 'label' => 'channel'];

	// tblsales_channel_detailt
	$fields['tblsales_channel_detailt_id']=['table_name' => 'sales_channel_detailt', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblsales_channel_detailt_customer_group']=['table_name' => 'customers_groups', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblsales_channel_detailt_customer']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];

	// tblwoocommere_store
	$fields['tblwoocommere_store_id']=['table_name' => 'woocommere_store', 'field_name' => 'id', 'label' => 'name'];

	 // tblwoocommere_store_detailt
	$fields['tblwoocommere_store_detailt_id']=['table_name' => 'woocommere_store_detailt', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblwoocommere_store_detailt_product_id']=['table_name' => 'items_groups', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblwoocommere_store_detailt_woocommere_store_id']=['table_name' => 'woocommere_store', 'field_name' => 'id', 'label' => 'name'];

	 // tblcart
	$fields['tblcart_id']=['table_name' => 'cart', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblcart_channel_id']=['table_name' => 'sales_channel', 'field_name' => 'id', 'label' => 'channel'];
	$fields['tblcart_userid']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblcart_number_invoice']=['table_name' => 'invoices', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblcart_seller']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];
	$fields['tblcart_warehouse_id']=['table_name' => 'warehouse', 'field_name' => 'id', 'label' => 'warehouse_name'];
	$fields['tblcart_shipping_country']=['table_name' => 'countries', 'field_name' => 'country_id', 'label' => 'short_name'];
	$fields['tblcart_status']=['table_name' => 'cart', 'field_name' => [0 => ['value' => 0, 'label' => _l('draft')], 1 => ['value' => 1, 'label' => _l('processing')], 2 => ['value' => 2, 'label' => _l('pending_payment')], 3 => ['value' => 3, 'label' => _l('confirm')],4 => ['value' => 4, 'label' => _l('shipping')],5 => ['value' => 5, 'label' => _l('fisnish')],6 => ['value' => 6, 'label' => _l('refund')],7 => ['value' => 7, 'label' => _l('lie')], 8 => ['value' => 8, 'label' => _l('cancelled')], ], 'label' => 'name'];


	 // tblcart_detailt
	$fields['tblcart_detailt_id']=['table_name' => 'cart_detailt', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblcart_detailt_product_id']=['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblcart_detailt_cart_id']=['table_name' => 'cart', 'field_name' => 'id', 'label' => 'name'];

	 // tblomni_trade_discount
	$fields['tblomni_trade_discount_id']=['table_name' => 'omni_trade_discount', 'field_name' => 'id', 'label' => 'name_trade_discount'];
	$fields['tblomni_trade_discount_group_clients']=['table_name' => 'customers_groups', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblomni_trade_discount_clients']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblomni_trade_discount_group_items']=['table_name' => 'items_groups', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblomni_trade_discount_items']=['table_name' => 'items', 'field_name' => 'id', 'label' => 'description'];
	$fields['tblomni_trade_discount_channel']=['table_name' => 'sales_channel', 'field_name' => 'id', 'label' => 'channel'];

	 // tblomni_log_sync_woo
	$fields['tblomni_log_sync_woo_id']=['table_name' => 'omni_log_sync_woo', 'field_name' => 'id', 'label' => 'name'];
	$fields['tblomni_log_sync_woo_order_id']=['table_name' => 'cart', 'field_name' => 'id', 'label' => 'name'];

	// tblomni_shift
	$fields['tblomni_shift_id']=['table_name' => 'omni_shift', 'field_name' => 'id', 'label' => 'shift_code'];
	$fields['tblomni_shift_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];


	// tblomni_shift_history
	$fields['tblomni_shift_history_id']=['table_name' => 'omni_shift_history', 'field_name' => 'id', 'label' => 'action'];
	$fields['tblomni_shift_history_shift_id']=['table_name' => 'omni_shift', 'field_name' => 'id', 'label' => 'shift_code'];
	$fields['tblomni_shift_history_customer_id']=['table_name' => 'clients', 'field_name' => 'userid', 'label' => 'company'];
	$fields['tblomni_shift_history_staff_id']=['table_name' => 'staff', 'field_name' => 'staffid', 'label' => 'firstname#lastname'];

	// tblomni_cart_payment
	$fields['tblomni_cart_payment_id']=['table_name' => 'omni_cart_payment', 'field_name' => 'id', 'label' => 'payment_name'];
	$fields['tblomni_cart_payment_cart_id']=['table_name' => 'cart', 'field_name' => 'id', 'label' => 'order_number'];
	$fields['tblomni_cart_payment_payment_id']=['table_name' => 'payment_modes', 'field_name' => 'id', 'label' => 'name'];

	// tblomni_pre_order_product_setting
	$fields['tblomni_pre_order_product_setting_id']=['table_name' => 'omni_pre_order_product_setting', 'field_name' => 'id', 'label' => 'id'];
	$fields['tblomni_pre_order_product_setting_channel_id']=['table_name' => 'sales_channel', 'field_name' => 'id', 'label' => 'channel'];
	$fields['tblomni_pre_order_product_setting_group_product_id']=['table_name' => 'items_groups', 'field_name' => 'id', 'label' => 'name'];












	return $fields;
}

/**
 * rb number field
 * @return [type] 
 */
function rb_number_field()
{
	// operator: = ; > ; < ; >= ; <= ; between ; != 
	//input type: input number
	$fields = [];

	//warehouse module
	$fields['tblgoods_receipt_detail_quantities'] = 'tblgoods_receipt_detail_quantities';
	$fields['tblgoods_receipt_detail_unit_price'] = 'tblgoods_receipt_detail_unit_price';
	$fields['tblgoods_receipt_detail_tax_money'] = 'tblgoods_receipt_detail_tax_money';
	$fields['tblgoods_receipt_detail_goods_money'] = 'tblgoods_receipt_detail_goods_money';
	$fields['tblgoods_receipt_detail_discount'] = 'tblgoods_receipt_detail_discount';
	$fields['tblgoods_receipt_detail_discount_money'] = 'tblgoods_receipt_detail_discount_money';
	$fields['tblgoods_receipt_detail_tax_rate'] = 'tblgoods_receipt_detail_tax_rate';
	$fields['tblgoods_receipt_total_tax_money'] = 'tblgoods_receipt_total_tax_money';
	$fields['tblgoods_receipt_total_goods_money'] = 'tblgoods_receipt_total_goods_money';
	$fields['tblgoods_receipt_value_of_inventory'] = 'tblgoods_receipt_value_of_inventory';
	$fields['tblgoods_receipt_total_money'] = 'tblgoods_receipt_total_money';

	//tblinvoices
	$fields['tblinvoices_subtotal'] = 'tblinvoices_subtotal';
	$fields['tblinvoices_total_tax'] = 'tblinvoices_total_tax';
	$fields['tblinvoices_total'] = 'tblinvoices_total';
	$fields['tblinvoices_m_taxrate'] = 'tblinvoices_m_taxrate';
	$fields['tblinvoices_m_taxname'] = 'tblinvoices_m_taxname';
	$fields['tblinvoices_total_amount'] = 'tblinvoices_total_amount';
	$fields['tblinvoices_adjustment'] = 'tblinvoices_adjustment';
	$fields['tblinvoices_discount_percent'] = 'tblinvoices_discount_percent';
	$fields['tblinvoices_discount_total'] = 'tblinvoices_discount_total';
	$fields['tblinvoices_discount_type'] = 'tblinvoices_discount_type';

	//tblestimates
	$fields['tblestimates_subtotal'] = 'tblestimates_subtotal';
	$fields['tblestimates_total_tax'] = 'tblestimates_total_tax';
	$fields['tblestimates_total'] = 'tblestimates_total';
	$fields['tblestimates_m_taxrate'] = 'tblestimates_m_taxrate';
	$fields['tblestimates_m_taxname'] = 'tblestimates_m_taxname';
	$fields['tblestimates_total_amount'] = 'tblestimates_total_amount';
	$fields['tblestimates_adjustment'] = 'tblestimates_adjustment';
	$fields['tblestimates_discount_percent'] = 'tblestimates_discount_percent';
	$fields['tblestimates_discount_total'] = 'tblestimates_discount_total';

	//tblcreditnotes
	$fields['tblcreditnotes_subtotal'] = 'tblcreditnotes_subtotal';
	$fields['tblcreditnotes_total_tax'] = 'tblcreditnotes_total_tax';
	$fields['tblcreditnotes_total'] = 'tblcreditnotes_total';
	$fields['tblcreditnotes_adjustment'] = 'tblcreditnotes_adjustment';
	$fields['tblcreditnotes_discount_percent'] = 'tblcreditnotes_discount_percent';
	$fields['tblcreditnotes_discount_total'] = 'tblcreditnotes_discount_total';

	//tbltasks
	$fields['tbltasks_hourly_rate'] = 'tbltasks_hourly_rate';

	//tblproposals
	$fields['tblproposals_total'] = 'tblproposals_total';
	$fields['tblproposals_subtotal'] = 'tblproposals_subtotal';
	$fields['tblproposals_total_tax'] = 'tblproposals_total_tax';
	$fields['tblproposals_m_taxrate'] = 'tblproposals_m_taxrate';
	$fields['tblproposals_m_taxname'] = 'tblproposals_m_taxname';
	$fields['tblproposals_total_amount'] = 'tblproposals_total_amount';
	$fields['tblproposals_adjustment'] = 'tblproposals_adjustment';
	$fields['tblproposals_discount_percent'] = 'tblproposals_discount_percent';
	$fields['tblproposals_discount_total'] = 'tblproposals_discount_total';

	//tblexpenses
	$fields['tblexpenses_amount'] = 'tblexpenses_amount';

	//tblleads
	$fields['tblleads_lead_value'] = 'tblleads_lead_value';

	//tblprojects
	$fields['tblprojects_project_cost'] = 'tblprojects_project_cost';
	$fields['tblprojects_project_rate_per_hour'] = 'tblprojects_project_rate_per_hour';
	$fields['tblprojects_estimated_hours'] = 'tblprojects_estimated_hours';

	//tblstaff
	$fields['tblstaff_hourly_rate'] = 'tblstaff_hourly_rate';

	//tblitem_tax
	$fields['tblitem_tax_taxrate'] = 'tblitem_tax_taxrate';

	//tblitemable
	$fields['tblitemable_qty'] = 'tblitemable_qty';
	$fields['tblitemable_rate'] = 'tblitemable_rate';
	$fields['tblitemable_wh_delivered_quantity'] = 'tblitemable_wh_delivered_quantity';

	//tblinvoicepaymentrecords
	$fields['tblinvoicepaymentrecords_amount'] = 'tblinvoicepaymentrecords_amount';

	//tblcredits
	$fields['tblcredits_amount'] = 'tblcredits_amount';

	//tblcreditnote_refunds
	$fields['tblcreditnote_refunds_amount'] = 'tblcreditnote_refunds_amount';

	//tblitems
	$fields['tblitems_rate'] = 'tblitems_rate';
	$fields['tblitems_purchase_price'] = 'tblitems_purchase_price';

	//tbltaxes
	$fields['tbltaxes_taxrate'] = 'tbltaxes_taxrate';

	//tblcontract_renewals
	$fields['tblcontract_renewals_old_value'] = 'tblcontract_renewals_old_value';
	$fields['tblcontract_renewals_new_value'] = 'tblcontract_renewals_new_value';

	//tbltaskstimers
	$fields['tbltaskstimers_hourly_rate'] = 'tbltaskstimers_hourly_rate';

	/*------------warehouse module-----------*/
	// tblwh_loss_adjustment_detail
	$fields['tblwh_loss_adjustment_detail_current_number'] = 'tblwh_loss_adjustment_detail_current_number';
	$fields['tblwh_loss_adjustment_detail_updates_number'] = 'tblwh_loss_adjustment_detail_updates_number';

	// tblgoods_delivery
	$fields['tblgoods_delivery_total_money'] = 'tblgoods_delivery_total_money';
	$fields['tblgoods_delivery_total_discount'] = 'tblgoods_delivery_total_discount';
	$fields['tblgoods_delivery_after_discount'] = 'tblgoods_delivery_after_discount';

	//tblgoods_delivery_detail
	$fields['tblgoods_delivery_detail_quantities'] = 'tblgoods_delivery_detail_quantities';
	$fields['tblgoods_delivery_detail_unit_price'] = 'tblgoods_delivery_detail_unit_price';
	$fields['tblgoods_delivery_detail_total_money'] = 'tblgoods_delivery_detail_total_money';
	$fields['tblgoods_delivery_detail_discount'] = 'tblgoods_delivery_detail_discount';
	$fields['tblgoods_delivery_detail_discount_money'] = 'tblgoods_delivery_detail_discount_money';
	$fields['tblgoods_delivery_detail_available_quantity'] = 'tblgoods_delivery_detail_available_quantity';
	$fields['tblgoods_delivery_detail_total_after_discount'] = 'tblgoods_delivery_detail_total_after_discount';

	//tblinternal_delivery_note
	$fields['tblinternal_delivery_note_total_amount'] = 'tblinternal_delivery_note_total_amount';

	//tblinternal_delivery_note_detail
	$fields['tblinternal_delivery_note_detail_available_quantity'] = 'tblinternal_delivery_note_detail_available_quantity';
	$fields['tblinternal_delivery_note_detail_quantities'] = 'tblinternal_delivery_note_detail_quantities';
	$fields['tblinternal_delivery_note_detail_unit_price'] = 'tblinternal_delivery_note_detail_unit_price';
	$fields['tblinternal_delivery_note_detail_into_money'] = 'tblinternal_delivery_note_detail_into_money';

	//tblinventory_manage
	$fields['tblinventory_manage_inventory_number'] = 'tblinventory_manage_inventory_number';

	// tblpur_debit_notes
	$fields['tblpur_debit_notes_subtotal'] = 'tblpur_debit_notes_subtotal';
	$fields['tblpur_debit_notes_total_tax'] = 'tblpur_debit_notes_total_tax';
	$fields['tblpur_debit_notes_total'] = 'tblpur_debit_notes_total';
	$fields['tblpur_debit_notes_adjustment'] = 'tblpur_debit_notes_adjustment';
	$fields['tblpur_debit_notes_discount_percent'] = 'tblpur_debit_notes_discount_percent';
	$fields['tblpur_debit_notes_discount_total'] = 'tblpur_debit_notes_discount_total';

	// tblpur_debits
	$fields['tblpur_debits_amount'] = 'tblpur_debits_amount';

	// tblpur_debits_refunds
	$fields['tblpur_debits_refunds_amount'] = 'tblpur_debits_refunds_amount';

	// tblpur_estimates
	$fields['tblpur_estimates_subtotal'] = 'tblpur_estimates_subtotal';
	$fields['tblpur_estimates_total_tax'] = 'tblpur_estimates_total_tax';
	$fields['tblpur_estimates_total'] = 'tblpur_estimates_total';
	$fields['tblpur_estimates_adjustment'] = 'tblpur_estimates_adjustment';
	$fields['tblpur_estimates_discount_percent'] = 'tblpur_estimates_discount_percent';
	$fields['tblpur_estimates_discount_total'] = 'tblpur_estimates_discount_total';

	// tblpur_estimate_detail
	$fields['tblpur_estimate_detail_unit_price'] = 'tblpur_estimate_detail_unit_price';
	$fields['tblpur_estimate_detail_quantity'] = 'tblpur_estimate_detail_quantity';
	$fields['tblpur_estimate_detail_into_money'] = 'tblpur_estimate_detail_into_money';
	$fields['tblpur_estimate_detail_total'] = 'tblpur_estimate_detail_total';
	$fields['tblpur_estimate_detail_total_money'] = 'tblpur_estimate_detail_total_money';
	$fields['tblpur_estimate_detail_discount_money'] = 'tblpur_estimate_detail_discount_money';
	$fields['tblpur_estimate_detail_discount_%'] = 'tblpur_estimate_detail_discount_%';
	$fields['tblpur_estimate_detail_tax_value'] = 'tblpur_estimate_detail_tax_value';

	// tblpur_invoices
	$fields['tblpur_invoices_subtotal'] = 'tblpur_invoices_subtotal';
	$fields['tblpur_invoices_tax'] = 'tblpur_invoices_tax';
	$fields['tblpur_invoices_total'] = 'tblpur_invoices_total';

	// tblpur_invoice_payment
	$fields['tblpur_invoice_payment_amount'] = 'tblpur_invoice_payment_amount';

	// tblpur_orders
	$fields['tblpur_orders_subtotal'] = 'tblpur_orders_subtotal';
	$fields['tblpur_orders_total_tax'] = 'tblpur_orders_total_tax';
	$fields['tblpur_orders_total'] = 'tblpur_orders_total';
	$fields['tblpur_orders_discount_percent'] = 'tblpur_orders_discount_percent';
	$fields['tblpur_orders_discount_total'] = 'tblpur_orders_discount_total';
	$fields['tblpur_orders_tax_order_rate'] = 'tblpur_orders_tax_order_rate';
	$fields['tblpur_orders_tax_order_amount'] = 'tblpur_orders_tax_order_amount';

	// tblpur_order_detail
	$fields['tblpur_order_detail_unit_price'] = 'tblpur_order_detail_unit_price';
	$fields['tblpur_order_detail_quantity'] = 'tblpur_order_detail_quantity';
	$fields['tblpur_order_detail_into_money'] = 'tblpur_order_detail_into_money';
	$fields['tblpur_order_detail_total'] = 'tblpur_order_detail_total';
	$fields['tblpur_order_detail_discount_%' ] = 'tblpur_order_detail_discount_%';
	$fields['tblpur_order_detail_discount_money'] = 'tblpur_order_detail_discount_money';
	$fields['tblpur_order_detail_total_money'] = 'tblpur_order_detail_total_money';
	$fields['tblpur_order_detail_tax_value'] = 'tblpur_order_detail_tax_value';

	// tblpur_request
	$fields['tblpur_request_subtotal'] = 'tblpur_request_subtotal';
	$fields['tblpur_request_total_tax'] = 'tblpur_request_total_tax';
	$fields['tblpur_request_total'] = 'tblpur_request_total';

	// tblpur_request_detail
	$fields['tblpur_request_detail_unit_price'] = 'tblpur_request_detail_unit_price';
	$fields['tblpur_request_detail_quantity'] = 'tblpur_request_detail_quantity';
	$fields['tblpur_request_detail_into_money'] = 'tblpur_request_detail_into_money';
	$fields['tblpur_request_detail_inventory_quantity'] = 'tblpur_request_detail_inventory_quantity';
	$fields['tblpur_request_detail_tax_rate'] = 'tblpur_request_detail_tax_rate';
	$fields['tblpur_request_detail_tax_value'] = 'tblpur_request_detail_tax_value';
	$fields['tblpur_request_detail_total'] = 'tblpur_request_detail_total';

	/*------omnisales module-------*/

	// tblsales_channel_detailt
	$fields['tblsales_channel_detailt_prices'] = 'tblsales_channel_detailt_prices';

	 // tblwoocommere_store_detailt
	$fields['tblwoocommere_store_detailt_prices'] = 'tblwoocommere_store_detailt_prices';

	 // tblcart
	$fields['tblcart_discount'] = 'tblcart_discount';
	$fields['tblcart_total'] = 'tblcart_total';
	$fields['tblcart_sub_total'] = 'tblcart_sub_total';
	$fields['tblcart_discount_total'] = 'tblcart_discount_total';
	$fields['tblcart_customers_pay'] = 'tblcart_customers_pay';
	$fields['tblcart_amount_returned'] = 'tblcart_amount_returned';
	$fields['tblcart_tax'] = 'tblcart_tax';
	$fields['tblcart_discount_percent'] = 'tblcart_discount_percent';
	$fields['tblcart_adjustment'] = 'tblcart_adjustment';
	$fields['tblcart_shipping_tax'] = 'tblcart_shipping_tax';

	 // tblcart_detailt
	$fields['tblcart_detailt_quantity'] = 'tblcart_detailt_quantity';
	$fields['tblcart_detailt_prices'] = 'tblcart_detailt_prices';
	$fields['tblcart_detailt_percent_discount'] = 'tblcart_detailt_percent_discount';
	$fields['tblcart_detailt_prices_discount'] = 'tblcart_detailt_prices_discount';

	 // tblomni_trade_discount
	$fields['tblomni_trade_discount_minimum_order_value'] = 'tblomni_trade_discount_minimum_order_value';

	 // tblomni_log_sync_woo
	$fields['tblomni_log_sync_woo_regular_price'] = 'tblomni_log_sync_woo_regular_price';
	$fields['tblomni_log_sync_woo_sale_price'] = 'tblomni_log_sync_woo_sale_price';
	$fields['tblomni_log_sync_woo_stock_quantity'] = 'tblomni_log_sync_woo_stock_quantity';
	$fields['tblomni_log_sync_woo_stock_quantity_history'] = 'tblomni_log_sync_woo_stock_quantity_history';

	// tblomni_shift
	$fields['tblomni_shift_granted_amount'] = 'tblomni_shift_granted_amount';
	$fields['tblomni_shift_incurred_amount'] = 'tblomni_shift_incurred_amount';
	$fields['tblomni_shift_closing_amount'] = 'tblomni_shift_closing_amount';
	$fields['tblomni_shift_order_value'] = 'tblomni_shift_order_value';

	// tblomni_shift_history
	$fields['tblomni_shift_history_granted_amount'] = 'tblomni_shift_history_granted_amount';
	$fields['tblomni_shift_history_current_amount'] = 'tblomni_shift_history_current_amount';
	$fields['tblomni_shift_history_customer_amount'] = 'tblomni_shift_history_customer_amount';
	$fields['tblomni_shift_history_balance_amount'] = 'tblomni_shift_history_balance_amount';
	$fields['tblomni_shift_history_order_value'] = 'tblomni_shift_history_order_value';

	// tblomni_cart_payment
	$fields['tblomni_cart_payment_customer_pay'] = 'tblomni_cart_payment_customer_pay';





	return $fields;

}

/**
 * rb text field
 * @return [type] 
 */
function rb_text_field()
{
	// operator: = ; like ; not like ; begin with ; end with ; IN ; NOT IN ; 
	//input type: input text
	$fields = [];

	//tblclients
	$fields['tblclients_company'] = 'tblclients_company';
	$fields['tblclients_vat'] = 'tblclients_vat';
	$fields['tblclients_phonenumber'] = 'tblclients_phonenumber';
	$fields['tblclients_city'] = 'tblclients_city';
	$fields['tblclients_zip'] = 'tblclients_zip';
	$fields['tblclients_state'] = 'tblclients_state';
	$fields['tblclients_address'] = 'tblclients_address';
	$fields['tblclients_website'] = 'tblclients_website';
	$fields['tblclients_billing_street'] = 'tblclients_billing_street';
	$fields['tblclients_billing_city'] = 'tblclients_billing_city';
	$fields['tblclients_billing_state'] = 'tblclients_billing_state';
	$fields['tblclients_billing_zip'] = 'tblclients_billing_zip';
	$fields['tblclients_billing_country'] = 'tblclients_billing_country';
	$fields['tblclients_shipping_street'] = 'tblclients_shipping_street';
	$fields['tblclients_shipping_city'] = 'tblclients_shipping_city';
	$fields['tblclients_shipping_state'] = 'tblclients_shipping_state';
	$fields['tblclients_shipping_zip'] = 'tblclients_shipping_zip';
	$fields['tblclients_shipping_country'] = 'tblclients_shipping_country';
	$fields['tblclients_longitude'] = 'tblclients_longitude';
	$fields['tblclients_latitude'] = 'tblclients_latitude';
	$fields['tblclients_default_language'] = 'tblclients_default_language';
	$fields['tblclients_show_primary_contact'] = 'tblclients_show_primary_contact';
	$fields['tblclients_stripe_id'] = 'tblclients_stripe_id';
	$fields['tblclients_registration_confirmed'] = 'tblclients_registration_confirmed';

	//tblcountries
	$fields['tblcountries_iso2'] = 'tblcountries_iso2';
	$fields['tblcountries_short_name'] = 'tblcountries_short_name';
	$fields['tblcountries_long_name'] = 'tblcountries_long_name';
	$fields['tblcountries_iso3'] = 'tblcountries_iso3';
	$fields['tblcountries_numcode'] = 'tblcountries_numcode';
	$fields['tblcountries_un_member'] = 'tblcountries_un_member';
	$fields['tblcountries_calling_code'] = 'tblcountries_calling_code';
	$fields['tblcountries_cctld'] = 'tblcountries_cctld';

	//tblcontact
	$fields['tblcontacts_is_primary'] = 'tblcontacts_is_primary';
	$fields['tblcontacts_firstname'] = 'tblcontacts_firstname';
	$fields['tblcontacts_lastname'] = 'tblcontacts_lastname';
	$fields['tblcontacts_email'] = 'tblcontacts_email';
	$fields['tblcontacts_phonenumber'] = 'tblcontacts_phonenumber';
	$fields['tblcontacts_title'] = 'tblcontacts_title';
	$fields['tblcontacts_password'] = 'tblcontacts_password';
	$fields['tblcontacts_new_pass_key'] = 'tblcontacts_new_pass_key';
	$fields['tblcontacts_new_pass_key_requested'] = 'tblcontacts_new_pass_key_requested';
	$fields['tblcontacts_email_verified_at'] = 'tblcontacts_email_verified_at';
	$fields['tblcontacts_email_verification_key'] = 'tblcontacts_email_verification_key';
	$fields['tblcontacts_email_verification_sent_at'] = 'tblcontacts_email_verification_sent_at';
	$fields['tblcontacts_last_ip'] = 'tblcontacts_last_ip';
	$fields['tblcontacts_last_login'] = 'tblcontacts_last_login';
	$fields['tblcontacts_last_password_change'] = 'tblcontacts_last_password_change';
	$fields['tblcontacts_profile_image'] = 'tblcontacts_profile_image';
	$fields['tblcontacts_direction'] = 'tblcontacts_direction';
	$fields['tblcontacts_invoice_emails'] = 'tblcontacts_invoice_emails';
	$fields['tblcontacts_estimate_emails'] = 'tblcontacts_estimate_emails';
	$fields['tblcontacts_credit_note_emails'] = 'tblcontacts_credit_note_emails';
	$fields['tblcontacts_contract_emails'] = 'tblcontacts_contract_emails';
	$fields['tblcontacts_task_emails'] = 'tblcontacts_task_emails';
	$fields['tblcontacts_project_emails'] = 'tblcontacts_project_emails';
	$fields['tblcontacts_ticket_emails'] = 'tblcontacts_ticket_emails';

	//tblcustomers_group
	$fields['tblcustomers_groups_name'] = 'tblcustomers_groups_name';
	
	//tblinvoices
	$fields['tblinvoices_deleted_customer_name'] = 'tblinvoices_deleted_customer_name';
	$fields['tblinvoices_number'] = 'tblinvoices_number';
	$fields['tblinvoices_prefix'] = 'tblinvoices_prefix';
	$fields['tblinvoices_number_format'] = 'tblinvoices_number_format';
	$fields['tblinvoices_hash'] = 'tblinvoices_hash';
	$fields['tblinvoices_clientnote'] = 'tblinvoices_clientnote';
	$fields['tblinvoices_adminnote'] = 'tblinvoices_adminnote';
	$fields['tblinvoices_cancel_overdue_reminders'] = 'tblinvoices_cancel_overdue_reminders';
	$fields['tblinvoices_allowed_payment_modes'] = 'tblinvoices_allowed_payment_modes';
	$fields['tblinvoices_token'] = 'tblinvoices_token';
	$fields['tblinvoices_recurring'] = 'tblinvoices_recurring';
	$fields['tblinvoices_recurring_type'] = 'tblinvoices_recurring_type';
	$fields['tblinvoices_custom_recurring'] = 'tblinvoices_custom_recurring';
	$fields['tblinvoices_cycles'] = 'tblinvoices_cycles';
	$fields['tblinvoices_total_cycles'] = 'tblinvoices_total_cycles';
	$fields['tblinvoices_is_recurring_from'] = 'tblinvoices_is_recurring_from';
	$fields['tblinvoices_terms'] = 'tblinvoices_terms';
	$fields['tblinvoices_billing_street'] = 'tblinvoices_billing_street';
	$fields['tblinvoices_billing_city'] = 'tblinvoices_billing_city';
	$fields['tblinvoices_billing_state'] = 'tblinvoices_billing_state';
	$fields['tblinvoices_billing_zip'] = 'tblinvoices_billing_zip';
	$fields['tblinvoices_billing_country'] = 'tblinvoices_billing_country';
	$fields['tblinvoices_shipping_street'] = 'tblinvoices_shipping_street';
	$fields['tblinvoices_shipping_city'] = 'tblinvoices_shipping_city';
	$fields['tblinvoices_shipping_state'] = 'tblinvoices_shipping_state';
	$fields['tblinvoices_shipping_zip'] = 'tblinvoices_shipping_zip';
	$fields['tblinvoices_shipping_country'] = 'tblinvoices_shipping_country';
	$fields['tblinvoices_include_shipping'] = 'tblinvoices_include_shipping';
	$fields['tblinvoices_show_shipping_on_invoice'] = 'tblinvoices_show_shipping_on_invoice';
	$fields['tblinvoices_show_quantity_as'] = 'tblinvoices_show_quantity_as';
	$fields['tblinvoices_short_link'] = 'tblinvoices_short_link';

	//tblestimates
	$fields['tblestimates_deleted_customer_name'] = 'tblestimates_deleted_customer_name';
	$fields['tblestimates_number'] = 'tblestimates_number';
	$fields['tblestimates_prefix'] = 'tblestimates_prefix';
	$fields['tblestimates_number_format'] = 'tblestimates_number_format';
	$fields['tblestimates_hash'] = 'tblestimates_hash';
	$fields['tblestimates_clientnote'] = 'tblestimates_clientnote';
	$fields['tblestimates_adminnote'] = 'tblestimates_adminnote';
	$fields['tblestimates_discount_type'] = 'tblestimates_discount_type';
	$fields['tblestimates_terms'] = 'tblestimates_terms';
	$fields['tblestimates_reference_no'] = 'tblestimates_reference_no';
	$fields['tblestimates_billing_street'] = 'tblestimates_billing_street';
	$fields['tblestimates_billing_city'] = 'tblestimates_billing_city';
	$fields['tblestimates_billing_state'] = 'tblestimates_billing_state';
	$fields['tblestimates_billing_zip'] = 'tblestimates_billing_zip';
	$fields['tblestimates_billing_country'] = 'tblestimates_billing_country';
	$fields['tblestimates_shipping_street'] = 'tblestimates_shipping_street';
	$fields['tblestimates_shipping_city'] = 'tblestimates_shipping_city';
	$fields['tblestimates_shipping_state'] = 'tblestimates_shipping_state';
	$fields['tblestimates_shipping_zip'] = 'tblestimates_shipping_zip';
	$fields['tblestimates_shipping_country'] = 'tblestimates_shipping_country';
	$fields['tblestimates_include_shipping'] = 'tblestimates_include_shipping';
	$fields['tblestimates_show_shipping_on_estimate'] = 'tblestimates_show_shipping_on_estimate';
	$fields['tblestimates_show_quantity_as'] = 'tblestimates_show_quantity_as';
	$fields['tblestimates_pipeline_order'] = 'tblestimates_pipeline_order';
	$fields['tblestimates_is_expiry_notified'] = 'tblestimates_is_expiry_notified';
	$fields['tblestimates_acceptance_firstname'] = 'tblestimates_acceptance_firstname';
	$fields['tblestimates_acceptance_lastname'] = 'tblestimates_acceptance_lastname';
	$fields['tblestimates_acceptance_email'] = 'tblestimates_acceptance_email';
	$fields['tblestimates_acceptance_date'] = 'tblestimates_acceptance_date';
	$fields['tblestimates_acceptance_ip'] = 'tblestimates_acceptance_ip';
	$fields['tblestimates_signature'] = 'tblestimates_signature';
	$fields['tblestimates_short_link'] = 'tblestimates_short_link';

	//tblcreditnotes
	$fields['tblcreditnotes_deleted_customer_name'] = 'tblcreditnotes_deleted_customer_name';
	$fields['tblcreditnotes_number'] = 'tblcreditnotes_number';
	$fields['tblcreditnotes_prefix'] = 'tblcreditnotes_prefix';
	$fields['tblcreditnotes_number_format'] = 'tblcreditnotes_number_format';
	$fields['tblcreditnotes_adminnote'] = 'tblcreditnotes_adminnote';
	$fields['tblcreditnotes_terms'] = 'tblcreditnotes_terms';
	$fields['tblcreditnotes_clientnote'] = 'tblcreditnotes_clientnote';
	$fields['tblcreditnotes_discount_type'] = 'tblcreditnotes_discount_type';
	$fields['tblcreditnotes_billing_street'] = 'tblcreditnotes_billing_street';
	$fields['tblcreditnotes_billing_city'] = 'tblcreditnotes_billing_city';
	$fields['tblcreditnotes_billing_state'] = 'tblcreditnotes_billing_state';
	$fields['tblcreditnotes_billing_zip'] = 'tblcreditnotes_billing_zip';
	$fields['tblcreditnotes_billing_country'] = 'tblcreditnotes_billing_country';
	$fields['tblcreditnotes_shipping_street'] = 'tblcreditnotes_shipping_street';
	$fields['tblcreditnotes_shipping_city'] = 'tblcreditnotes_shipping_city';
	$fields['tblcreditnotes_shipping_state'] = 'tblcreditnotes_shipping_state';
	$fields['tblcreditnotes_shipping_zip'] = 'tblcreditnotes_shipping_zip';
	$fields['tblcreditnotes_shipping_country'] = 'tblcreditnotes_shipping_country';
	$fields['tblcreditnotes_include_shipping'] = 'tblcreditnotes_include_shipping';
	$fields['tblcreditnotes_show_shipping_on_credit_note'] = 'tblcreditnotes_show_shipping_on_credit_note';
	$fields['tblcreditnotes_show_quantity_as'] = 'tblcreditnotes_show_quantity_as';
	$fields['tblcreditnotes_reference_no'] = 'tblcreditnotes_reference_no';

	//tbltickets
	$fields['tbltickets_email'] = 'tbltickets_email';
	$fields['tbltickets_name'] = 'tbltickets_name';
	$fields['tbltickets_priority'] = 'tbltickets_priority';
	$fields['tbltickets_service'] = 'tbltickets_service';
	$fields['tbltickets_ticketkey'] = 'tbltickets_ticketkey';
	$fields['tbltickets_subject'] = 'tbltickets_subject';
	$fields['tbltickets_message'] = 'tbltickets_message';
	$fields['tbltickets_admin'] = 'tbltickets_admin';
	$fields['tbltickets_lastreply'] = 'tbltickets_lastreply';
	$fields['tbltickets_clientread'] = 'tbltickets_clientread';
	$fields['tbltickets_adminread'] = 'tbltickets_adminread';

	//subscriptions
	$fields['tblsubscriptions_name'] = 'tblsubscriptions_name';
	$fields['tblsubscriptions_description'] = 'tblsubscriptions_description';
	$fields['tblsubscriptions_description_in_item'] = 'tblsubscriptions_description_in_item';
	$fields['tblsubscriptions_terms'] = 'tblsubscriptions_terms';
	$fields['tblsubscriptions_stripe_tax_id'] = 'tblsubscriptions_stripe_tax_id';
	$fields['tblsubscriptions_tax_id_2'] = 'tblsubscriptions_tax_id_2';
	$fields['tblsubscriptions_stripe_tax_id_2'] = 'tblsubscriptions_stripe_tax_id_2';
	$fields['tblsubscriptions_stripe_plan_id'] = 'tblsubscriptions_stripe_plan_id';
	$fields['tblsubscriptions_stripe_subscription_id'] = 'tblsubscriptions_stripe_subscription_id';
	$fields['tblsubscriptions_next_billing_cycle'] = 'tblsubscriptions_next_billing_cycle';
	$fields['tblsubscriptions_ends_at'] = 'tblsubscriptions_ends_at';
	$fields['tblsubscriptions_quantity'] = 'tblsubscriptions_quantity';
	$fields['tblsubscriptions_hash'] = 'tblsubscriptions_hash';
	$fields['tblsubscriptions_in_test_environment'] = 'tblsubscriptions_in_test_environment';

	//tblcontracts
	$fields['tblcontracts_content'] = 'tblcontracts_content';
	$fields['tblcontracts_description'] = 'tblcontracts_description';
	$fields['tblcontracts_subject'] = 'tblcontracts_subject';
	$fields['tblcontracts_isexpirynotified'] = 'tblcontracts_isexpirynotified';
	$fields['tblcontracts_contract_value'] = 'tblcontracts_contract_value';
	$fields['tblcontracts_trash'] = 'tblcontracts_trash';
	$fields['tblcontracts_not_visible_to_client'] = 'tblcontracts_not_visible_to_client';
	$fields['tblcontracts_hash'] = 'tblcontracts_hash';
	$fields['tblcontracts_signed'] = 'tblcontracts_signed';
	$fields['tblcontracts_signature'] = 'tblcontracts_signature';
	$fields['tblcontracts_marked_as_signed'] = 'tblcontracts_marked_as_signed';
	$fields['tblcontracts_acceptance_firstname'] = 'tblcontracts_acceptance_firstname';
	$fields['tblcontracts_acceptance_lastname'] = 'tblcontracts_acceptance_lastname';
	$fields['tblcontracts_acceptance_email'] = 'tblcontracts_acceptance_email';
	$fields['tblcontracts_acceptance_ip'] = 'tblcontracts_acceptance_ip';
	$fields['tblcontracts_short_link'] = 'tblcontracts_short_link';

	//tbltasks
	$fields['tbltasks_name'] = 'tbltasks_name';
	$fields['tbltasks_description'] = 'tbltasks_description';
	$fields['tbltasks_priority'] = 'tbltasks_priority';
	$fields['tbltasks_is_added_from_contact'] = 'tbltasks_is_added_from_contact';
	$fields['tbltasks_recurring_type'] = 'tbltasks_recurring_type';
	$fields['tbltasks_repeat_every'] = 'tbltasks_repeat_every';
	$fields['tbltasks_recurring'] = 'tbltasks_recurring';
	$fields['tbltasks_is_recurring_from'] = 'tbltasks_is_recurring_from';
	$fields['tbltasks_cycles'] = 'tbltasks_cycles';
	$fields['tbltasks_total_cycles'] = 'tbltasks_total_cycles';
	$fields['tbltasks_custom_recurring'] = 'tbltasks_custom_recurring';
	$fields['tbltasks_rel_id'] = 'tbltasks_rel_id';
	$fields['tbltasks_rel_type'] = 'tbltasks_rel_type';
	$fields['tbltasks_is_public'] = 'tbltasks_is_public';
	$fields['tbltasks_billable'] = 'tbltasks_billable';
	$fields['tbltasks_billed'] = 'tbltasks_billed';
	$fields['tbltasks_milestone'] = 'tbltasks_milestone';
	$fields['tbltasks_kanban_order'] = 'tbltasks_kanban_order';
	$fields['tbltasks_milestone_order'] = 'tbltasks_milestone_order';
	$fields['tbltasks_visible_to_client'] = 'tbltasks_visible_to_client';
	$fields['tbltasks_deadline_notified'] = 'tbltasks_deadline_notified';
	$fields['tbltasks_checklist_templates_id'] = 'tbltasks_checklist_templates_id';
	$fields['tbltasks_checklist_templates_description'] = 'tbltasks_checklist_templates_description';


	//tbltask_comments
	$fields['tbltask_comments_content'] = 'tbltask_comments_content';

	//tblreminders
	$fields['tblreminders_description'] = 'tblreminders_description';
	$fields['tblreminders_isnotified'] = 'tblreminders_isnotified';
	$fields['tblreminders_rel_id'] = 'tblreminders_rel_id';
	$fields['tblreminders_rel_type'] = 'tblreminders_rel_type';
	$fields['tblreminders_notify_by_email'] = 'tblreminders_notify_by_email';

	//tblproposals
	$fields['tblproposals_subject'] = 'tblproposals_subject';
	$fields['tblproposals_content'] = 'tblproposals_content';
	$fields['tblproposals_discount_type'] = 'tblproposals_discount_type';
	$fields['tblproposals_show_quantity_as'] = 'tblproposals_show_quantity_as';
	$fields['tblproposals_currency'] = 'tblproposals_currency';
	$fields['tblproposals_rel_id'] = 'tblproposals_rel_id';
	$fields['tblproposals_rel_type'] = 'tblproposals_rel_type';
	$fields['tblproposals_hash'] = 'tblproposals_hash';
	$fields['tblproposals_proposal_to'] = 'tblproposals_proposal_to';
	$fields['tblproposals_zip'] = 'tblproposals_zip';
	$fields['tblproposals_state'] = 'tblproposals_state';
	$fields['tblproposals_city'] = 'tblproposals_city';
	$fields['tblproposals_address'] = 'tblproposals_address';
	$fields['tblproposals_email'] = 'tblproposals_email';
	$fields['tblproposals_phone'] = 'tblproposals_phone';
	$fields['tblproposals_allow_comments'] = 'tblproposals_allow_comments';
	$fields['tblproposals_pipeline_order'] = 'tblproposals_pipeline_order';
	$fields['tblproposals_is_expiry_notified'] = 'tblproposals_is_expiry_notified';
	$fields['tblproposals_acceptance_firstname'] = 'tblproposals_acceptance_firstname';
	$fields['tblproposals_acceptance_lastname'] = 'tblproposals_acceptance_lastname';
	$fields['tblproposals_acceptance_email'] = 'tblproposals_acceptance_email';
	$fields['tblproposals_acceptance_ip'] = 'tblproposals_acceptance_ip';
	$fields['tblproposals_signature'] = 'tblproposals_signature';
	$fields['tblproposals_short_link'] = 'tblproposals_short_link';
	$fields['tblproposals_processing'] = 'tblproposals_processing';

	//tblexpenses
	$fields['tblexpenses_category'] = 'tblexpenses_category';
	$fields['tblexpenses_tax'] = 'tblexpenses_tax';
	$fields['tblexpenses_tax2'] = 'tblexpenses_tax2';
	$fields['tblexpenses_reference_no'] = 'tblexpenses_reference_no';
	$fields['tblexpenses_note'] = 'tblexpenses_note';
	$fields['tblexpenses_expense_name'] = 'tblexpenses_expense_name';
	$fields['tblexpenses_billable'] = 'tblexpenses_billable';
	$fields['tblexpenses_invoiceid'] = 'tblexpenses_invoiceid';
	$fields['tblexpenses_paymentmode'] = 'tblexpenses_paymentmode';
	$fields['tblexpenses_recurring_type'] = 'tblexpenses_recurring_type';
	$fields['tblexpenses_repeat_every'] = 'tblexpenses_repeat_every';
	$fields['tblexpenses_recurring'] = 'tblexpenses_recurring';
	$fields['tblexpenses_cycles'] = 'tblexpenses_cycles';
	$fields['tblexpenses_total_cycles'] = 'tblexpenses_total_cycles';
	$fields['tblexpenses_custom_recurring'] = 'tblexpenses_custom_recurring';
	$fields['tblexpenses_create_invoice_billable'] = 'tblexpenses_create_invoice_billable';
	$fields['tblexpenses_send_invoice_to_customer'] = 'tblexpenses_send_invoice_to_customer';
	$fields['tblexpenses_recurring_from'] = 'tblexpenses_recurring_from';
	$fields['tblexpenses_vendor'] = 'tblexpenses_vendor';


	//tblleads
	$fields['tblleads_hash'] = 'tblleads_hash';
	$fields['tblleads_name'] = 'tblleads_name';
	$fields['tblleads_title'] = 'tblleads_title';
	$fields['tblleads_company'] = 'tblleads_company';
	$fields['tblleads_description'] = 'tblleads_description';
	$fields['tblleads_zip'] = 'tblleads_zip';
	$fields['tblleads_city'] = 'tblleads_city';
	$fields['tblleads_state'] = 'tblleads_state';
	$fields['tblleads_address'] = 'tblleads_address';
	$fields['tblleads_dateadded'] = 'tblleads_dateadded';
	$fields['tblleads_from_form_id'] = 'tblleads_from_form_id';
	$fields['tblleads_source'] = 'tblleads_source';
	$fields['tblleads_email'] = 'tblleads_email';
	$fields['tblleads_website'] = 'tblleads_website';
	$fields['tblleads_leadorder'] = 'tblleads_leadorder';
	$fields['tblleads_phonenumber'] = 'tblleads_phonenumber';
	$fields['tblleads_lost'] = 'tblleads_lost';
	$fields['tblleads_junk'] = 'tblleads_junk';
	$fields['tblleads_last_lead_status'] = 'tblleads_last_lead_status';
	$fields['tblleads_is_imported_from_email_integration'] = 'tblleads_is_imported_from_email_integration';
	$fields['tblleads_email_integration_uid'] = 'tblleads_email_integration_uid';
	$fields['tblleads_is_public'] = 'tblleads_is_public';
	$fields['tblleads_default_language'] = 'tblleads_default_language';
	$fields['tblleads_vat'] = 'tblleads_vat';
	$fields['tblleads_status'] = 'tblleads_status';

	//tblprojects
	$fields['tblprojects_name'] = 'tblprojects_name';
	$fields['tblprojects_description'] = 'tblprojects_description';
	$fields['tblprojects_billing_type'] = 'tblprojects_billing_type';
	$fields['tblprojects_progress'] = 'tblprojects_progress';
	$fields['tblprojects_progress_from_tasks'] = 'tblprojects_progress_from_tasks';
	$fields['tblprojects_contact_notification'] = 'tblprojects_contact_notification';
	$fields['tblprojects_notify_contacts'] = 'tblprojects_notify_contacts';

	//tblprojectdiscussions
	$fields['tblprojectdiscussions_subject'] = 'tblprojectdiscussions_subject';
	$fields['tblprojectdiscussions_description'] = 'tblprojectdiscussions_description';
	$fields['tblprojectdiscussions_show_to_customer'] = 'tblprojectdiscussions_show_to_customer';

	//tblprojectdiscussioncomments
	$fields['tblprojectdiscussioncomments_discussion_type'] = 'tblprojectdiscussioncomments_discussion_type';
	$fields['tblprojectdiscussioncomments_parent'] = 'tblprojectdiscussioncomments_parent';
	$fields['tblprojectdiscussioncomments_content'] = 'tblprojectdiscussioncomments_content';
	$fields['tblprojectdiscussioncomments_fullname'] = 'tblprojectdiscussioncomments_fullname';
	$fields['tblprojectdiscussioncomments_file_name'] = 'tblprojectdiscussioncomments_file_name';
	$fields['tblprojectdiscussioncomments_file_mime_type'] = 'tblprojectdiscussioncomments_file_mime_type';

	//tblstaff
	$fields['tblstaff_email'] = 'tblstaff_email';
	$fields['tblstaff_firstname'] = 'tblstaff_firstname';
	$fields['tblstaff_lastname'] = 'tblstaff_lastname';
	$fields['tblstaff_facebook'] = 'tblstaff_facebook';
	$fields['tblstaff_linkedin'] = 'tblstaff_linkedin';
	$fields['tblstaff_phonenumber'] = 'tblstaff_phonenumber';
	$fields['tblstaff_skype'] = 'tblstaff_skype';
	$fields['tblstaff_password'] = 'tblstaff_password';
	$fields['tblstaff_profile_image'] = 'tblstaff_profile_image';
	$fields['tblstaff_last_ip'] = 'tblstaff_last_ip';
	$fields['tblstaff_new_pass_key'] = 'tblstaff_new_pass_key';
	$fields['tblstaff_admin'] = 'tblstaff_admin';
	$fields['tblstaff_active'] = 'tblstaff_active';
	$fields['tblstaff_default_language'] = 'tblstaff_default_language';
	$fields['tblstaff_direction'] = 'tblstaff_direction';
	$fields['tblstaff_media_path_slug'] = 'tblstaff_media_path_slug';
	$fields['tblstaff_is_not_staff'] = 'tblstaff_is_not_staff';
	$fields['tblstaff_two_factor_auth_enabled'] = 'tblstaff_two_factor_auth_enabled';
	$fields['tblstaff_two_factor_auth_code'] = 'tblstaff_two_factor_auth_code';
	$fields['tblstaff_two_factor_auth_code_requested'] = 'tblstaff_two_factor_auth_code_requested';
	$fields['tblstaff_email_signature'] = 'tblstaff_email_signature';
	$fields['tblstaff_birthplace'] = 'tblstaff_birthplace';
	$fields['tblstaff_sex'] = 'tblstaff_sex';
	$fields['tblstaff_marital_status'] = 'tblstaff_marital_status';
	$fields['tblstaff_nation'] = 'tblstaff_nation';
	$fields['tblstaff_religion'] = 'tblstaff_religion';
	$fields['tblstaff_identification'] = 'tblstaff_identification';
	$fields['tblstaff_home_town'] = 'tblstaff_home_town';
	$fields['tblstaff_resident'] = 'tblstaff_resident';
	$fields['tblstaff_current_address'] = 'tblstaff_current_address';
	$fields['tblstaff_literacy'] = 'tblstaff_literacy';
	$fields['tblstaff_orther_infor'] = 'tblstaff_orther_infor';
	$fields['tblstaff_job_position'] = 'tblstaff_job_position';
	$fields['tblstaff_workplace'] = 'tblstaff_workplace';
	$fields['tblstaff_place_of_issue'] = 'tblstaff_place_of_issue';
	$fields['tblstaff_account_number'] = 'tblstaff_account_number';
	$fields['tblstaff_name_account'] = 'tblstaff_name_account';
	$fields['tblstaff_issue_bank'] = 'tblstaff_issue_bank';
	$fields['tblstaff_records_received'] = 'tblstaff_records_received';
	$fields['tblstaff_Personal_tax_code'] = 'tblstaff_Personal_tax_code';
	$fields['tblstaff_google_auth_secret'] = 'tblstaff_google_auth_secret';
	$fields['tblstaff_staff_identifi'] = 'tblstaff_staff_identifi';

	//tblproposal_comments
	$fields['tblproposal_comments_content'] = 'tblproposal_comments_content';

	//tblitem_tax
	$fields['tblitem_tax_rel_id'] = 'tblitem_tax_rel_id';
	$fields['tblitem_tax_rel_type'] = 'tblitem_tax_rel_type';
	$fields['tblitem_tax_taxname'] = 'tblitem_tax_taxname';

	//tblitem_tax
	$fields['tblitemable_rel_id'] = 'tblitemable_rel_id';
	$fields['tblitemable_rel_type'] = 'tblitemable_rel_type';
	$fields['tblitemable_description'] = 'tblitemable_description';
	$fields['tblitemable_long_description'] = 'tblitemable_long_description';
	$fields['tblitemable_unit'] = 'tblitemable_unit';
	$fields['tblitemable_item_order'] = 'tblitemable_item_order';

	//tblcurrencies
	$fields['tblcurrencies_symbol'] = 'tblcurrencies_symbol';
	$fields['tblcurrencies_name'] = 'tblcurrencies_name';
	$fields['tblcurrencies_decimal_separator'] = 'tblcurrencies_decimal_separator';
	$fields['tblcurrencies_thousand_separator'] = 'tblcurrencies_thousand_separator';
	$fields['tblcurrencies_placement'] = 'tblcurrencies_placement';
	$fields['tblcurrencies_isdefault'] = 'tblcurrencies_isdefault';

	//tblnotes
	$fields['tblnotes_rel_id'] = 'tblnotes_rel_id';
	$fields['tblnotes_rel_type'] = 'tblnotes_rel_type';
	$fields['tblnotes_description'] = 'tblnotes_description';

	//tblinvoicepaymentrecords
	$fields['tblinvoicepaymentrecords_paymentmode'] = 'tblinvoicepaymentrecords_paymentmode';
	$fields['tblinvoicepaymentrecords_paymentmethod'] = 'tblinvoicepaymentrecords_paymentmethod';
	$fields['tblinvoicepaymentrecords_note'] = 'tblinvoicepaymentrecords_note';
	$fields['tblinvoicepaymentrecords_transactionid'] = 'tblinvoicepaymentrecords_transactionid';

	//tblrelated_items
	$fields['tblrelated_items_rel_id'] = 'tblrelated_items_rel_id';
	$fields['tblrelated_items_rel_type'] = 'tblrelated_items_rel_type';

	//tblpayment_modes
	$fields['tblpayment_modes_name'] = 'tblpayment_modes_name';
	$fields['tblpayment_modes_description'] = 'tblpayment_modes_description';
	$fields['tblpayment_modes_show_on_pdf'] = 'tblpayment_modes_show_on_pdf';
	$fields['tblpayment_modes_invoices_only'] = 'tblpayment_modes_invoices_only';
	$fields['tblpayment_modes_expenses_only'] = 'tblpayment_modes_expenses_only';
	$fields['tblpayment_modes_selected_by_default'] = 'tblpayment_modes_selected_by_default';
	$fields['tblpayment_modes_active'] = 'tblpayment_modes_active';

	//tblcreditnote_refunds
	$fields['tblcreditnote_refunds_payment_mode'] = 'tblcreditnote_refunds_payment_mode';
	$fields['tblcreditnote_refunds_note'] = 'tblcreditnote_refunds_note';

	//tblitems
	$fields['tblitems_description'] = 'tblitems_description';
	$fields['tblitems_long_description'] = 'tblitems_long_description';
	$fields['tblitems_unit'] = 'tblitems_unit';
	$fields['tblitems_commodity_code'] = 'tblitems_commodity_code';
	$fields['tblitems_commodity_barcode'] = 'tblitems_commodity_barcode';
	$fields['tblitems_commodity_type'] = 'tblitems_commodity_type';
	$fields['tblitems_warehouse_id'] = 'tblitems_warehouse_id';
	$fields['tblitems_origin'] = 'tblitems_origin';
	$fields['tblitems_sku_code'] = 'tblitems_sku_code';
	$fields['tblitems_sku_name'] = 'tblitems_sku_name';
	$fields['tblitems_commodity_name'] = 'tblitems_commodity_name';
	$fields['tblitems_color'] = 'tblitems_color';
	$fields['tblitems_guarantee'] = 'tblitems_guarantee';
	$fields['tblitems_profif_ratio'] = 'tblitems_profif_ratio';
	$fields['tblitems_active'] = 'tblitems_active';
	$fields['tblitems_long_descriptions'] = 'tblitems_long_descriptions';
	$fields['tblitems_without_checking_warehouse'] = 'tblitems_without_checking_warehouse';
	$fields['tblitems_series_id'] = 'tblitems_series_id';
	$fields['tblitems_parent_id'] = 'tblitems_parent_id';
	$fields['tblitems_attributes'] = 'tblitems_attributes';
	$fields['tblitems_parent_attributes'] = 'tblitems_parent_attributes';
	$fields['tblitems_product_type'] = 'tblitems_product_type';
	$fields['tblitems_purchase_unit_measure'] = 'tblitems_purchase_unit_measure';
	$fields['tblitems_can_be_sold'] = 'tblitems_can_be_sold';
	$fields['tblitems_can_be_purchased'] = 'tblitems_can_be_purchased';
	$fields['tblitems_can_be_manufacturing'] = 'tblitems_can_be_manufacturing';
	$fields['tblitems_invoice_policy'] = 'tblitems_invoice_policy';
	$fields['tblitems_description_sale'] = 'tblitems_description_sale';
	$fields['tblitems_supplier_taxes_id'] = 'tblitems_supplier_taxes_id';
	$fields['tblitems_replenish_on_order'] = 'tblitems_replenish_on_order';
	$fields['tblitems_manufacture'] = 'tblitems_manufacture';
	$fields['tblitems_manufacturing_lead_time'] = 'tblitems_manufacturing_lead_time';
	$fields['tblitems_customer_lead_time'] = 'tblitems_customer_lead_time';
	$fields['tblitems_weight'] = 'tblitems_weight';
	$fields['tblitems_volume'] = 'tblitems_volume';
	$fields['tblitems_hs_code'] = 'tblitems_hs_code';
	$fields['tblitems_description_delivery_orders'] = 'tblitems_description_delivery_orders';
	$fields['tblitems_description_receipts'] = 'tblitems_description_receipts';
	$fields['tblitems_description_internal_transfers'] = 'tblitems_description_internal_transfers';

	//tblitems_groups
	$fields['tblitems_groups_name'] = 'tblitems_groups_name';
	$fields['tblitems_groups_commodity_group_code'] = 'tblitems_groups_commodity_group_code';
	$fields['tblitems_groups_order'] = 'tblitems_groups_order';
	$fields['tblitems_groups_display'] = 'tblitems_groups_display';
	$fields['tblitems_groups_note'] = 'tblitems_groups_note';

	//tbltaxes
	$fields['tbltaxes_taxrate'] = 'tbltaxes_taxrate';

	//tblexpenses_categories
	$fields['tblexpenses_categories_name'] = 'tblexpenses_categories_name';
	$fields['tblexpenses_categories_description'] = 'tblexpenses_categories_description';

	//tblcontracts_types
	$fields['tblcontracts_types_name'] = 'tblcontracts_types_name';

	//tblcontract_comments
	$fields['tblcontract_comments_content'] = 'tblcontract_comments_content';

	//tblcontract_renewals
	$fields['tblcontract_renewals_renewed_by'] = 'tblcontract_renewals_renewed_by';
	$fields['tblcontract_renewals_is_on_old_expiry_notified'] = 'tblcontract_renewals_is_on_old_expiry_notified';

	//tbltaskstimers
	$fields['tbltaskstimers_start_time'] = 'tbltaskstimers_start_time';
	$fields['tbltaskstimers_end_time'] = 'tbltaskstimers_end_time';
	$fields['tbltaskstimers_note'] = 'tbltaskstimers_note';

	//tblproject_settings
	$fields['tblproject_settings_name'] = 'tblproject_settings_name';
	$fields['tblproject_settings_value'] = 'tblproject_settings_value';

	//tblmilestones
	$fields['tblmilestones_name'] = 'tblmilestones_name';
	$fields['tblmilestones_description'] = 'tblmilestones_description';
	$fields['tblmilestones_description_visible_to_customer'] = 'tblmilestones_description_visible_to_customer';
	$fields['tblmilestones_color'] = 'tblmilestones_color';
	$fields['tblmilestones_milestone_order'] = 'tblmilestones_milestone_order';

	//tblproject_notes
	$fields['tblproject_notes_content'] = 'tblproject_notes_content';


	/*-------------warehouse module-------------*/
	$fields['tblgoods_receipt_supplier_code'] = 'tblgoods_receipt_supplier_code';
	$fields['tblgoods_receipt_supplier_name'] = 'tblgoods_receipt_supplier_name';
	$fields['tblgoods_receipt_deliver_name'] = 'tblgoods_receipt_deliver_name';
	$fields['tblgoods_receipt_description'] = 'tblgoods_receipt_description';
	$fields['tblgoods_receipt_goods_receipt_code'] = 'tblgoods_receipt_goods_receipt_code';
	$fields['tblgoods_receipt_invoice_no'] = 'tblgoods_receipt_invoice_no';
	$fields['tblgoods_receipt_detail_note'] = 'tblgoods_receipt_detail_note';
	$fields['tblgoods_receipt_detail_lot_number'] = 'tblgoods_receipt_detail_lot_number';

	// tblwh_loss_adjustment
	$fields['tblwh_loss_adjustment_reason'] = 'tblwh_loss_adjustment_reason';
	$fields['tblwh_loss_adjustment_type'] = 'tblwh_loss_adjustment_type';

	// tblwh_loss_adjustment_detail
	$fields['tblwh_loss_adjustment_detail_expiry_date'] = 'tblwh_loss_adjustment_detail_expiry_date';
	$fields['tblwh_loss_adjustment_detail_lot_number'] = 'tblwh_loss_adjustment_detail_lot_number';
	$fields['tblwh_loss_adjustment_reason'] = 'tblwh_loss_adjustment_reason';

	// tblgoods_delivery
	$fields['tblgoods_delivery_rel_type'] = 'tblgoods_delivery_rel_type';
	$fields['tblgoods_delivery_rel_document'] = 'tblgoods_delivery_rel_document';
	$fields['tblgoods_delivery_customer_name'] = 'tblgoods_delivery_customer_name';
	$fields['tblgoods_delivery_to_'] = 'tblgoods_delivery_to_';
	$fields['tblgoods_delivery_address'] = 'tblgoods_delivery_address';
	$fields['tblgoods_delivery_description'] = 'tblgoods_delivery_description';
	$fields['tblgoods_delivery_goods_delivery_code'] = 'tblgoods_delivery_goods_delivery_code';
	$fields['tblgoods_delivery_warehouse_id'] = 'tblgoods_delivery_warehouse_id';
	$fields['tblgoods_delivery_type'] = 'tblgoods_delivery_type';
	$fields['tblgoods_delivery_invoice_no'] = 'tblgoods_delivery_invoice_no';
	$fields['tblgoods_delivery_type_of_delivery'] = 'tblgoods_delivery_type_of_delivery';

	//tblgoods_delivery_detail
	$fields['tblgoods_delivery_detail_commodity_name'] = 'tblgoods_delivery_detail_commodity_name';
	$fields['tblgoods_delivery_detail_note'] = 'tblgoods_delivery_detail_note';
	$fields['tblgoods_delivery_detail_tax_id'] = 'tblgoods_delivery_detail_tax_id';
	$fields['tblgoods_delivery_detail_lot_number'] = 'tblgoods_delivery_detail_lot_number';
	$fields['tblgoods_delivery_detail_guarantee_period'] = 'tblgoods_delivery_detail_guarantee_period';

	//tblinternal_delivery_note
	$fields['tblinternal_delivery_note_internal_delivery_name'] = 'tblinternal_delivery_note_internal_delivery_name';
	$fields['tblinternal_delivery_note_description'] = 'tblinternal_delivery_note_description';
	$fields['tblinternal_delivery_note_internal_delivery_code'] = 'tblinternal_delivery_note_internal_delivery_code';

	//tblinternal_delivery_note_detail
	$fields['tblinternal_delivery_note_detail_note'] = 'tblinternal_delivery_note_detail_note';

	//tblwarehouse
	$fields['tblwarehouse_warehouse_code'] = 'tblwarehouse_warehouse_code';
	$fields['tblwarehouse_warehouse_name'] = 'tblwarehouse_warehouse_name';
	$fields['tblwarehouse_warehouse_address'] = 'tblwarehouse_warehouse_address';
	$fields['tblwarehouse_order'] = 'tblwarehouse_order';
	$fields['tblwarehouse_display'] = 'tblwarehouse_display';
	$fields['tblwarehouse_note'] = 'tblwarehouse_note';
	$fields['tblwarehouse_city'] = 'tblwarehouse_city';
	$fields['tblwarehouse_state'] = 'tblwarehouse_state';
	$fields['tblwarehouse_zip_code'] = 'tblwarehouse_zip_code';

	//tblware_commodity_type
	$fields['tblware_commodity_type_commondity_code'] = 'tblware_commodity_type_commondity_code';
	$fields['tblware_commodity_type_commondity_name'] = 'tblware_commodity_type_commondity_name';
	$fields['tblware_commodity_type_order'] = 'tblware_commodity_type_order';
	$fields['tblware_commodity_type_display'] = 'tblware_commodity_type_display';
	$fields['tblware_commodity_type_note'] = 'tblware_commodity_type_note';

	//tblwh_sub_group
	$fields['tblwh_sub_group_sub_group_code'] = 'tblwh_sub_group_sub_group_code';
	$fields['tblwh_sub_group_sub_group_name'] = 'tblwh_sub_group_sub_group_name';
	$fields['tblwh_sub_group_order'] = 'tblwh_sub_group_order';
	$fields['tblwh_sub_group_display'] = 'tblwh_sub_group_display';
	$fields['tblwh_sub_group_note'] = 'tblwh_sub_group_note';

	//tblware_unit_type
	$fields['tblware_unit_type_unit_code'] = 'tblware_unit_type_unit_code';
	$fields['tblware_unit_type_unit_name'] = 'tblware_unit_type_unit_name';
	$fields['tblware_unit_type_unit_symbol'] = 'tblware_unit_type_unit_symbol';
	$fields['tblware_unit_type_order'] = 'tblware_unit_type_order';
	$fields['tblware_unit_type_display'] = 'tblware_unit_type_display';
	$fields['tblware_unit_type_note'] = 'tblware_unit_type_note';
	$fields['tblware_unit_type_category_id'] = 'tblware_unit_type_category_id';
	$fields['tblware_unit_type_unit_measure_type'] = 'tblware_unit_type_unit_measure_type';
	$fields['tblware_unit_type_bigger_ratio'] = 'tblware_unit_type_bigger_ratio';
	$fields['tblware_unit_type_smaller_ratio'] = 'tblware_unit_type_smaller_ratio';
	$fields['tblware_unit_type_rounding'] = 'tblware_unit_type_rounding';

	//tblware_color
	$fields['tblware_color_color_code'] = 'tblware_color_color_code';
	$fields['tblware_color_color_name'] = 'tblware_color_color_name';
	$fields['tblware_color_color_hex'] = 'tblware_color_color_hex';
	$fields['tblware_color_order'] = 'tblware_color_order';
	$fields['tblware_color_display'] = 'tblware_color_display';
	$fields['tblware_color_note'] = 'tblware_color_note';

	//tblware_body_type
	$fields['tblware_body_type_body_code'] = 'tblware_body_type_body_code';
	$fields['tblware_body_type_body_name'] = 'tblware_body_type_body_name';
	$fields['tblware_body_type_order'] = 'tblware_body_type_order';
	$fields['tblware_body_type_display'] = 'tblware_body_type_display';
	$fields['tblware_body_type_note'] = 'tblware_body_type_note';

	//tblware_size_type
	$fields['tblware_size_type_size_code'] = 'tblware_size_type_size_code';
	$fields['tblware_size_type_size_name'] = 'tblware_size_type_size_name';
	$fields['tblware_size_type_size_symbol'] = 'tblware_size_type_size_symbol';
	$fields['tblware_size_type_order'] = 'tblware_size_type_order';
	$fields['tblware_size_type_display'] = 'tblware_size_type_display';
	$fields['tblware_size_type_note'] = 'tblware_size_type_note';

	//tblware_style_type
	$fields['tblware_style_type_style_code'] = 'tblware_style_type_style_code';
	$fields['tblware_style_type_style_barcode'] = 'tblware_style_type_style_barcode';
	$fields['tblware_style_type_style_name'] = 'tblware_style_type_style_name';
	$fields['tblware_style_type_order'] = 'tblware_style_type_order';
	$fields['tblware_style_type_display'] = 'tblware_style_type_display';
	$fields['tblware_style_type_note'] = 'tblware_style_type_note';

	//tblinventory_manage
	$fields['tblinventory_manage_lot_number'] = 'tblinventory_manage_lot_number';


	// tblpur_comments
	$fields['tblpur_comments_content'] = 'tblpur_comments_content';
	$fields['tblpur_comments_rel_type'] = 'tblpur_comments_rel_type';
	$fields['tblpur_comments_rel_id'] = 'tblpur_comments_rel_id';

	// tblpur_contacts
	$fields['tblpur_contacts_is_primary'] = 'tblpur_contacts_is_primary';
	$fields['tblpur_contacts_firstname'] = 'tblpur_contacts_firstname';
	$fields['tblpur_contacts_lastname'] = 'tblpur_contacts_lastname';
	$fields['tblpur_contacts_email'] = 'tblpur_contacts_email';
	$fields['tblpur_contacts_phonenumber'] = 'tblpur_contacts_phonenumber';
	$fields['tblpur_contacts_title'] = 'tblpur_contacts_title';
	$fields['tblpur_contacts_password'] = 'tblpur_contacts_password';
	$fields['tblpur_contacts_new_pass_key'] = 'tblpur_contacts_new_pass_key';
	$fields['tblpur_contacts_new_pass_key_requested'] = 'tblpur_contacts_new_pass_key_requested';
	$fields['tblpur_contacts_email_verified_at'] = 'tblpur_contacts_email_verified_at';
	$fields['tblpur_contacts_email_verification_key'] = 'tblpur_contacts_email_verification_key';
	$fields['tblpur_contacts_email_verification_sent_at'] = 'tblpur_contacts_email_verification_sent_at';
	$fields['tblpur_contacts_last_ip'] = 'tblpur_contacts_last_ip';
	$fields['tblpur_contacts_active'] = 'tblpur_contacts_active';
	$fields['tblpur_contacts_profile_image'] = 'tblpur_contacts_profile_image';
	$fields['tblpur_contacts_direction'] = 'tblpur_contacts_direction';
	$fields['tblpur_contacts_invoice_emails'] = 'tblpur_contacts_invoice_emails';
	$fields['tblpur_contacts_estimate_emails'] = 'tblpur_contacts_estimate_emails';
	$fields['tblpur_contacts_credit_note_emails'] = 'tblpur_contacts_credit_note_emails';
	$fields['tblpur_contacts_contract_emails'] = 'tblpur_contacts_contract_emails';
	$fields['tblpur_contacts_task_emails'] = 'tblpur_contacts_task_emails';
	$fields['tblpur_contacts_project_emails'] = 'tblpur_contacts_project_emails';
	$fields['tblpur_contacts_ticket_emails'] = 'tblpur_contacts_ticket_emails';

	//tblpur_contracts
	$fields['tblpur_contracts_contract_number'] = 'tblpur_contracts_contract_number';
	$fields['tblpur_contracts_contract_name'] = 'tblpur_contracts_contract_name';
	$fields['tblpur_contracts_content'] = 'tblpur_contracts_content';
	$fields['tblpur_contracts_contract_value'] = 'tblpur_contracts_contract_value';
	$fields['tblpur_contracts_time_payment'] = 'tblpur_contracts_time_payment';
	$fields['tblpur_contracts_note'] = 'tblpur_contracts_note';
	$fields['tblpur_contracts_signed_status'] = 'tblpur_contracts_signed_status';
	$fields['tblpur_contracts_service_category'] = 'tblpur_contracts_service_category';
	$fields['tblpur_contracts_payment_terms'] = 'tblpur_contracts_payment_terms';
	$fields['tblpur_contracts_payment_amount'] = 'tblpur_contracts_payment_amount';
	$fields['tblpur_contracts_payment_cycle'] = 'tblpur_contracts_payment_cycle';

	// tblpur_debit_notes
	$fields['tblpur_debit_notes_deleted_vendor_name'] = 'tblpur_debit_notes_deleted_vendor_name';
	$fields['tblpur_debit_notes_number'] = 'tblpur_debit_notes_number';
	$fields['tblpur_debit_notes_prefix'] = 'tblpur_debit_notes_prefix';
	$fields['tblpur_debit_notes_number_format'] = 'tblpur_debit_notes_number_format';
	$fields['tblpur_debit_notes_adminnote'] = 'tblpur_debit_notes_adminnote';
	$fields['tblpur_debit_notes_terms'] = 'tblpur_debit_notes_terms';
	$fields['tblpur_debit_notes_vendornote'] = 'tblpur_debit_notes_vendornote';
	$fields['tblpur_debit_notes_currency'] = 'tblpur_debit_notes_currency';
	$fields['tblpur_debit_notes_status'] = 'tblpur_debit_notes_status';
	$fields['tblpur_debit_notes_discount_type'] = 'tblpur_debit_notes_discount_type';
	$fields['tblpur_debit_notes_billing_street'] = 'tblpur_debit_notes_billing_street';
	$fields['tblpur_debit_notes_billing_city'] = 'tblpur_debit_notes_billing_city';
	$fields['tblpur_debit_notes_billing_state'] = 'tblpur_debit_notes_billing_state';
	$fields['tblpur_debit_notes_billing_zip'] = 'tblpur_debit_notes_billing_zip';
	$fields['tblpur_debit_notes_billing_country'] = 'tblpur_debit_notes_billing_country';
	$fields['tblpur_debit_notes_shipping_street'] = 'tblpur_debit_notes_shipping_street';
	$fields['tblpur_debit_notes_shipping_city'] = 'tblpur_debit_notes_shipping_city';
	$fields['tblpur_debit_notes_shipping_state'] = 'tblpur_debit_notes_shipping_state';
	$fields['tblpur_debit_notes_shipping_zip'] = 'tblpur_debit_notes_shipping_zip';
	$fields['tblpur_debit_notes_shipping_country'] = 'tblpur_debit_notes_shipping_country';
	$fields['tblpur_debit_notes_include_shipping'] = 'tblpur_debit_notes_include_shipping';
	$fields['tblpur_debit_notes_show_shipping_on_debit_note'] = 'tblpur_debit_notes_show_shipping_on_debit_note';
	$fields['tblpur_debit_notes_show_quantity_as'] = 'tblpur_debit_notes_show_quantity_as';
	$fields['tblpur_debit_notes_reference_no'] = 'tblpur_debit_notes_reference_no';

	// tblpur_debits_refunds
	$fields['tblpur_debits_refunds_payment_mode'] = 'tblpur_debits_refunds_payment_mode';
	$fields['tblpur_debits_refunds_note'] = 'tblpur_debits_refunds_note';

	// tblpur_estimates
	$fields['tblpur_estimates_sent'] = 'tblpur_estimates_sent';
	$fields['tblpur_estimates_deleted_vendor_name'] = 'tblpur_estimates_deleted_vendor_name';
	$fields['tblpur_estimates_number'] = 'tblpur_estimates_number';
	$fields['tblpur_estimates_prefix'] = 'tblpur_estimates_prefix';
	$fields['tblpur_estimates_number_format'] = 'tblpur_estimates_number_format';
	$fields['tblpur_estimates_hash'] = 'tblpur_estimates_hash';
	$fields['tblpur_estimates_currency'] = 'tblpur_estimates_currency';
	$fields['tblpur_estimates_addedfrom'] = 'tblpur_estimates_addedfrom';
	$fields['tblpur_estimates_status'] = 'tblpur_estimates_status';
	$fields['tblpur_estimates_vendornote'] = 'tblpur_estimates_vendornote';
	$fields['tblpur_estimates_adminnote'] = 'tblpur_estimates_adminnote';
	$fields['tblpur_estimates_discount_type'] = 'tblpur_estimates_discount_type';
	$fields['tblpur_estimates_invoiceid'] = 'tblpur_estimates_invoiceid';
	$fields['tblpur_estimates_terms'] = 'tblpur_estimates_terms';
	$fields['tblpur_estimates_reference_no'] = 'tblpur_estimates_reference_no';
	$fields['tblpur_estimates_billing_street'] = 'tblpur_estimates_billing_street';
	$fields['tblpur_estimates_billing_city'] = 'tblpur_estimates_billing_city';
	$fields['tblpur_estimates_billing_state'] = 'tblpur_estimates_billing_state';
	$fields['tblpur_estimates_billing_zip'] = 'tblpur_estimates_billing_zip';
	$fields['tblpur_estimates_billing_country'] = 'tblpur_estimates_billing_country';
	$fields['tblpur_estimates_shipping_street'] = 'tblpur_estimates_shipping_street';
	$fields['tblpur_estimates_shipping_city'] = 'tblpur_estimates_shipping_city';
	$fields['tblpur_estimates_shipping_state'] = 'tblpur_estimates_shipping_state';
	$fields['tblpur_estimates_shipping_zip'] = 'tblpur_estimates_shipping_zip';
	$fields['tblpur_estimates_shipping_country'] = 'tblpur_estimates_shipping_country';
	$fields['tblpur_estimates_include_shipping'] = 'tblpur_estimates_include_shipping';
	$fields['tblpur_estimates_show_shipping_on_estimate'] = 'tblpur_estimates_show_shipping_on_estimate';
	$fields['tblpur_estimates_show_quantity_as'] = 'tblpur_estimates_show_quantity_as';
	$fields['tblpur_estimates_pipeline_order'] = 'tblpur_estimates_pipeline_order';
	$fields['tblpur_estimates_is_expiry_notified'] = 'tblpur_estimates_is_expiry_notified';
	$fields['tblpur_estimates_acceptance_firstname'] = 'tblpur_estimates_acceptance_firstname';
	$fields['tblpur_estimates_acceptance_lastname'] = 'tblpur_estimates_acceptance_lastname';
	$fields['tblpur_estimates_acceptance_email'] = 'tblpur_estimates_acceptance_email';
	$fields['tblpur_estimates_acceptance_ip'] = 'tblpur_estimates_acceptance_ip';
	$fields['tblpur_estimates_signature'] = 'tblpur_estimates_signature';

	// tblpur_estimate_detail
	$fields['tblpur_estimate_detail_tax'] = 'tblpur_estimate_detail_tax';
	$fields['tblpur_estimate_detail_tax_rate'] = 'tblpur_estimate_detail_tax_rate';

	// tblpur_invoices
	$fields['tblpur_invoices_number'] = 'tblpur_invoices_number';
	$fields['tblpur_invoices_invoice_number'] = 'tblpur_invoices_invoice_number';
	$fields['tblpur_invoices_tax_rate'] = 'tblpur_invoices_tax_rate';
	$fields['tblpur_invoices_transactionid'] = 'tblpur_invoices_transactionid';
	$fields['tblpur_invoices_payment_request_status'] = 'tblpur_invoices_payment_request_status';
	$fields['tblpur_invoices_payment_status'] = 'tblpur_invoices_payment_status';
	$fields['tblpur_invoices_vendor_note'] = 'tblpur_invoices_vendor_note';
	$fields['tblpur_invoices_adminnote'] = 'tblpur_invoices_adminnote';
	$fields['tblpur_invoices_terms'] = 'tblpur_invoices_terms';
	
	// tblpur_invoice_payment
	$fields['tblpur_invoice_payment_paymentmode'] = 'tblpur_invoice_payment_paymentmode';
	$fields['tblpur_invoice_payment_note'] = 'tblpur_invoice_payment_note';
	$fields['tblpur_invoice_payment_transactionid'] = 'tblpur_invoice_payment_transactionid';
	$fields['tblpur_invoice_payment_approval_status'] = 'tblpur_invoice_payment_approval_status';

	// tblpur_orders
	$fields['tblpur_orders_pur_order_name'] = 'tblpur_orders_pur_order_name';
	$fields['tblpur_orders_pur_order_number'] = 'tblpur_orders_pur_order_number';
	$fields['tblpur_orders_status'] = 'tblpur_orders_status';
	$fields['tblpur_orders_days_owed'] = 'tblpur_orders_days_owed';
	$fields['tblpur_orders_addedfrom'] = 'tblpur_orders_addedfrom';
	$fields['tblpur_orders_vendornote'] = 'tblpur_orders_vendornote';
	$fields['tblpur_orders_terms'] = 'tblpur_orders_terms';
	$fields['tblpur_orders_discount_type'] = 'tblpur_orders_discount_type';
	$fields['tblpur_orders_status_goods'] = 'tblpur_orders_status_goods';
	$fields['tblpur_orders_number'] = 'tblpur_orders_number';
	$fields['tblpur_orders_expense_convert'] = 'tblpur_orders_expense_convert';
	$fields['tblpur_orders_hash'] = 'tblpur_orders_hash';
	$fields['tblpur_orders_delivery_status'] = 'tblpur_orders_delivery_status';
	$fields['tblpur_orders_type'] = 'tblpur_orders_type';


	// tblpur_order_detail
	$fields['tblpur_request_pur_rq_code'] = 'tblpur_request_pur_rq_code';
	$fields['tblpur_request_pur_rq_name'] = 'tblpur_request_pur_rq_name';
	$fields['tblpur_request_rq_description'] = 'tblpur_request_rq_description';
	$fields['tblpur_request_status'] = 'tblpur_request_status';
	$fields['tblpur_request_status_goods'] = 'tblpur_request_status_goods';
	$fields['tblpur_request_hash'] = 'tblpur_request_hash';
	$fields['tblpur_request_type'] = 'tblpur_request_type';
	$fields['tblpur_request_number'] = 'tblpur_request_number';
	$fields['tblpur_request_from_items'] = 'tblpur_request_from_items';

	// tblpur_order_detail
	$fields['tblpur_order_detail_description'] = 'tblpur_order_detail_description';
	$fields['tblpur_order_detail_tax'] = 'tblpur_order_detail_tax';
	$fields['tblpur_order_detail_tax_rate'] = 'tblpur_order_detail_tax_rate';
	$fields['tblpur_order_detail_wh_quantity_received'] = 'tblpur_order_detail_wh_quantity_received';

	// tblpur_request_detail
	$fields['tblpur_request_detail_item_text'] = 'tblpur_request_detail_item_text';
	$fields['tblpur_request_detail_tax'] = 'tblpur_request_detail_tax';

	// tblpur_vendor
	$fields['tblpur_vendor_company'] = 'tblpur_vendor_company';
	$fields['tblpur_vendor_vat'] = 'tblpur_vendor_vat';
	$fields['tblpur_vendor_phonenumber'] = 'tblpur_vendor_phonenumber';
	$fields['tblpur_vendor_country'] = 'tblpur_vendor_country';
	$fields['tblpur_vendor_city'] = 'tblpur_vendor_city';
	$fields['tblpur_vendor_zip'] = 'tblpur_vendor_zip';
	$fields['tblpur_vendor_state'] = 'tblpur_vendor_state';
	$fields['tblpur_vendor_address'] = 'tblpur_vendor_address';
	$fields['tblpur_vendor_website'] = 'tblpur_vendor_website';
	$fields['tblpur_vendor_active'] = 'tblpur_vendor_active';
	$fields['tblpur_vendor_leadid'] = 'tblpur_vendor_leadid';
	$fields['tblpur_vendor_billing_street'] = 'tblpur_vendor_billing_street';
	$fields['tblpur_vendor_billing_city'] = 'tblpur_vendor_billing_city';
	$fields['tblpur_vendor_billing_state'] = 'tblpur_vendor_billing_state';
	$fields['tblpur_vendor_billing_zip'] = 'tblpur_vendor_billing_zip';
	$fields['tblpur_vendor_billing_country'] = 'tblpur_vendor_billing_country';
	$fields['tblpur_vendor_shipping_street'] = 'tblpur_vendor_shipping_street';
	$fields['tblpur_vendor_shipping_city'] = 'tblpur_vendor_shipping_city';
	$fields['tblpur_vendor_shipping_state'] = 'tblpur_vendor_shipping_state';
	$fields['tblpur_vendor_shipping_zip'] = 'tblpur_vendor_shipping_zip';
	$fields['tblpur_vendor_shipping_country'] = 'tblpur_vendor_shipping_country';
	$fields['tblpur_vendor_longitude'] = 'tblpur_vendor_longitude';
	$fields['tblpur_vendor_latitude'] = 'tblpur_vendor_latitude';
	$fields['tblpur_vendor_default_language'] = 'tblpur_vendor_default_language';
	$fields['tblpur_vendor_default_currency'] = 'tblpur_vendor_default_currency';
	$fields['tblpur_vendor_show_primary_contact'] = 'tblpur_vendor_show_primary_contact';
	$fields['tblpur_vendor_stripe_id'] = 'tblpur_vendor_stripe_id';
	$fields['tblpur_vendor_registration_confirmed'] = 'tblpur_vendor_registration_confirmed';
	$fields['tblpur_vendor_bank_detail'] = 'tblpur_vendor_bank_detail';
	$fields['tblpur_vendor_payment_terms'] = 'tblpur_vendor_payment_terms';
	$fields['tblpur_vendor_vendor_code'] = 'tblpur_vendor_vendor_code';

	// tblpur_vendor_cate
	$fields['tblpur_vendor_cate_category_name'] = 'tblpur_vendor_cate_category_name';
	$fields['tblpur_vendor_cate_description'] = 'tblpur_vendor_cate_description';


	/*--------Omnisalse module-----*/

	// tblsales_channel
	$fields['tblsales_channel_channel'] = 'tblsales_channel_channel';
	$fields['tblsales_channel_status'] = 'tblsales_channel_status';

	// tblsales_channel_detailt
	$fields['tblsales_channel_detailt_group_product_id'] = 'tblsales_channel_detailt_group_product_id';
	$fields['tblsales_channel_detailt_product_id'] = 'tblsales_channel_detailt_product_id';
	$fields['tblsales_channel_detailt_sales_channel_id'] = 'tblsales_channel_detailt_sales_channel_id';
	$fields['tblsales_channel_detailt_department'] = 'tblsales_channel_detailt_department';
	$fields['tblsales_channel_detailt_pre_order_product_st_id'] = 'tblsales_channel_detailt_pre_order_product_st_id';

	// tblwoocommere_store
	$fields['tblwoocommere_store_name'] = 'tblwoocommere_store_name';
	$fields['tblwoocommere_store_ip'] = 'tblwoocommere_store_ip';
	$fields['tblwoocommere_store_url'] = 'tblwoocommere_store_url';
	$fields['tblwoocommere_store_port'] = 'tblwoocommere_store_port';
	$fields['tblwoocommere_store_token'] = 'tblwoocommere_store_token';

	 // tblwoocommere_store_detailt
	$fields['tblwoocommere_store_detailt_group_product_id'] = 'tblwoocommere_store_detailt_group_product_id';

	 // tblcart
	$fields['tblcart_id_contact'] = 'tblcart_id_contact';
	$fields['tblcart_name'] = 'tblcart_name';
	$fields['tblcart_address'] = 'tblcart_address';
	$fields['tblcart_phone_number'] = 'tblcart_phone_number';
	$fields['tblcart_voucher'] = 'tblcart_voucher';
	$fields['tblcart_status'] = 'tblcart_status';
	$fields['tblcart_complete'] = 'tblcart_complete';
	$fields['tblcart_order_number'] = 'tblcart_order_number';
	$fields['tblcart_channel'] = 'tblcart_channel';
	$fields['tblcart_first_name'] = 'tblcart_first_name';
	$fields['tblcart_last_name'] = 'tblcart_last_name';
	$fields['tblcart_email'] = 'tblcart_email';
	$fields['tblcart_company'] = 'tblcart_company';
	$fields['tblcart_phonenumber'] = 'tblcart_phonenumber';
	$fields['tblcart_city'] = 'tblcart_city';
	$fields['tblcart_state'] = 'tblcart_state';
	$fields['tblcart_country'] = 'tblcart_country';
	$fields['tblcart_zip'] = 'tblcart_zip';
	$fields['tblcart_billing_street'] = 'tblcart_billing_street';
	$fields['tblcart_billing_city'] = 'tblcart_billing_city';
	$fields['tblcart_billing_state'] = 'tblcart_billing_state';
	$fields['tblcart_billing_country'] = 'tblcart_billing_country';
	$fields['tblcart_billing_zip'] = 'tblcart_billing_zip';
	$fields['tblcart_shipping_street'] = 'tblcart_shipping_street';
	$fields['tblcart_shipping_city'] = 'tblcart_shipping_city';
	$fields['tblcart_shipping_state'] = 'tblcart_shipping_state';
	$fields['tblcart_shipping_zip'] = 'tblcart_shipping_zip';
	$fields['tblcart_notes'] = 'tblcart_notes';
	$fields['tblcart_reason'] = 'tblcart_reason';
	$fields['tblcart_admin_action'] = 'tblcart_admin_action';
	$fields['tblcart_discount_type'] = 'tblcart_discount_type';
	$fields['tblcart_invoice'] = 'tblcart_invoice';
	$fields['tblcart_stock_export_number'] = 'tblcart_stock_export_number';
	$fields['tblcart_create_invoice'] = 'tblcart_create_invoice';
	$fields['tblcart_stock_export'] = 'tblcart_stock_export';
	$fields['tblcart_staff_note'] = 'tblcart_staff_note';
	$fields['tblcart_payment_note'] = 'tblcart_payment_note';
	$fields['tblcart_allowed_payment_modes'] = 'tblcart_allowed_payment_modes';
	$fields['tblcart_shipping'] = 'tblcart_shipping';
	$fields['tblcart_payment_method_title'] = 'tblcart_payment_method_title';
	$fields['tblcart_discount_type_str'] = 'tblcart_discount_type_str';
	$fields['tblcart_currency'] = 'tblcart_currency';
	$fields['tblcart_terms'] = 'tblcart_terms';
	$fields['tblcart_enable'] = 'tblcart_enable';
	$fields['tblcart_shipping_tax_json'] = 'tblcart_shipping_tax_json';

	 // tblcart_detailt
	$fields['tblcart_detailt_classify'] = 'tblcart_detailt_classify';
	$fields['tblcart_detailt_product_name'] = 'tblcart_detailt_product_name';
	$fields['tblcart_detailt_long_description'] = 'tblcart_detailt_long_description';
	$fields['tblcart_detailt_sku'] = 'tblcart_detailt_sku';
	$fields['tblcart_detailt_tax'] = 'tblcart_detailt_tax';

	 // tblomni_trade_discount
	$fields['tblomni_trade_discount_name_trade_discount'] = 'tblomni_trade_discount_name_trade_discount';
	$fields['tblomni_trade_discount_formal'] = 'tblomni_trade_discount_formal';
	$fields['tblomni_trade_discount_discount'] = 'tblomni_trade_discount_discount';
	$fields['tblomni_trade_discount_voucher'] = 'tblomni_trade_discount_voucher';
	$fields['tblomni_trade_discount_store'] = 'tblomni_trade_discount_store';

	 // tblomni_log_sync_woo
	$fields['tblomni_log_sync_woo_name'] = 'tblomni_log_sync_woo_name';
	$fields['tblomni_log_sync_woo_short_description'] = 'tblomni_log_sync_woo_short_description';
	$fields['tblomni_log_sync_woo_sku'] = 'tblomni_log_sync_woo_sku';
	$fields['tblomni_log_sync_woo_type'] = 'tblomni_log_sync_woo_type';
	$fields['tblomni_log_sync_woo_chanel'] = 'tblomni_log_sync_woo_chanel';
	$fields['tblomni_log_sync_woo_company'] = 'tblomni_log_sync_woo_company';
	$fields['tblomni_log_sync_woo_description'] = 'tblomni_log_sync_woo_description';

	// tblomni_shift
	$fields['tblomni_shift_shift_code'] = 'tblomni_shift_shift_code';
	$fields['tblomni_shift_status'] = 'tblomni_shift_status';

	// tblomni_shift_history
	$fields['tblomni_shift_history_action'] = 'tblomni_shift_history_action';
	$fields['tblomni_shift_history_type'] = 'tblomni_shift_history_type';

	// tblomni_cart_payment
	$fields['tblomni_cart_payment_payment_name'] = 'tblomni_cart_payment_payment_name';

	// tblomni_pre_order_product_setting
	$fields['tblomni_pre_order_product_setting_customer_group'] = 'tblomni_pre_order_product_setting_customer_group';
	$fields['tblomni_pre_order_product_setting_customer'] = 'tblomni_pre_order_product_setting_customer';



	return $fields;
}

/**
 * rb_date_field
 * @return [type] 
 */
function rb_date_field()
{
	// operator: between 
	//input type: input date (two input)
	$fields = [];

	//tblinvoices
	$fields['tblinvoices_date'] = 'tblinvoices_date';
	$fields['tblinvoices_duedate'] = 'tblinvoices_duedate';
	$fields['tblinvoices_last_overdue_reminder'] = 'tblinvoices_last_overdue_reminder';
	$fields['tblinvoices_last_due_reminder'] = 'tblinvoices_last_due_reminder';
	$fields['tblinvoices_last_recurring_date'] = 'tblinvoices_last_recurring_date';

	//tblestimates
	
	$fields['tblestimates_date'] = 'tblestimates_date';
	$fields['tblestimates_expirydate'] = 'tblestimates_expirydate';


	//warehouse module
	$fields['tblgoods_receipt_date_c'] = 'tblgoods_receipt_date_c';
	$fields['tblgoods_receipt_date_add'] = 'tblgoods_receipt_date_add';
	$fields['tblgoods_receipt_expiry_date'] = 'tblgoods_receipt_expiry_date';
	$fields['tblgoods_receipt_detail_date_manufacture'] = 'tblgoods_receipt_detail_date_manufacture';
	$fields['tblgoods_receipt_detail_expiry_date'] = 'tblgoods_receipt_detail_expiry_date';

	//tblcreditnotes
	$fields['tblcreditnotes_date'] = 'tblcreditnotes_date';

	//subscriptions
	$fields['tblsubscriptions_created'] = 'tblsubscriptions_created';
	$fields['tblsubscriptions_date_subscribed'] = 'tblsubscriptions_date_subscribed';

	//tblcontracts
	$fields['tblcontracts_datestart'] = 'tblcontracts_datestart';
	$fields['tblcontracts_dateend'] = 'tblcontracts_dateend';

	//tbltasks
	$fields['tbltasks_startdate'] = 'tbltasks_startdate';
	$fields['tbltasks_duedate'] = 'tbltasks_duedate';
	$fields['tbltasks_last_recurring_date'] = 'tbltasks_last_recurring_date';

	//tblproposals
	$fields['tblproposals_open_till'] = 'tblproposals_open_till';
	$fields['tblproposals_date'] = 'tblproposals_date';

	//tblexpenses
	$fields['tblexpenses_date'] = 'tblexpenses_date';
	$fields['tblexpenses_last_recurring_date'] = 'tblexpenses_last_recurring_date';

	//tblleads
	$fields['tblleads_dateassigned'] = 'tblleads_dateassigned';

	//tblprojects
	$fields['tblprojects_start_date'] = 'tblprojects_start_date';
	$fields['tblprojects_deadline'] = 'tblprojects_deadline';
	$fields['tblprojects_project_created'] = 'tblprojects_project_created';

	//tblstaff
	$fields['tblstaff_birthday'] = 'tblstaff_birthday';
	$fields['tblstaff_days_for_identity'] = 'tblstaff_days_for_identity';
	$fields['tblstaff_date_update'] = 'tblstaff_date_update';

	//tblinvoicepaymentrecords
	$fields['tblinvoicepaymentrecords_date'] = 'tblinvoicepaymentrecords_date';

	//tblcredits
	$fields['tblcredits_date'] = 'tblcredits_date';

	//tblcreditnote_refunds
	$fields['tblcreditnote_refunds_refunded_on'] = 'tblcreditnote_refunds_refunded_on';

	//tblcontract_renewals
	$fields['tblcontract_renewals_old_start_date'] = 'tblcontract_renewals_old_start_date';
	$fields['tblcontract_renewals_new_start_date'] = 'tblcontract_renewals_new_start_date';
	$fields['tblcontract_renewals_old_end_date'] = 'tblcontract_renewals_old_end_date';
	$fields['tblcontract_renewals_new_end_date'] = 'tblcontract_renewals_new_end_date';

	//tblmilestones
	$fields['tblmilestones_due_date'] = 'tblmilestones_due_date';
	$fields['tblmilestones_datecreated'] = 'tblmilestones_datecreated';

	//tbltask_assigned
	$fields['tbltask_assigned_is_assigned_from_contact'] = 'tbltask_assigned_is_assigned_from_contact';



	/*-------------warehousem module----------*/
	// tblwh_loss_adjustment
	$fields['tblwh_loss_adjustment_time'] = 'tblwh_loss_adjustment_time';

	// tblgoods_delivery
	$fields['tblgoods_delivery_date_c'] = 'tblgoods_delivery_date_c';
	$fields['tblgoods_delivery_date_add'] = 'tblgoods_delivery_date_add';

	//tblgoods_delivery_detail
	$fields['tblgoods_delivery_detail_expiry_date'] = 'tblgoods_delivery_detail_expiry_date';

	//tblinternal_delivery_note
	$fields['tblinternal_delivery_note_date_c'] = 'tblinternal_delivery_note_date_c';
	$fields['tblinternal_delivery_note_date_add'] = 'tblinternal_delivery_note_date_add';

	//tblinventory_manage
	$fields['tblinventory_manage_date_manufacture'] = 'tblinventory_manage_date_manufacture';
	$fields['tblinventory_manage_expiry_date'] = 'tblinventory_manage_expiry_date';

	//tblpur_contracts
	$fields['tblpur_contracts_start_date'] = 'tblpur_contracts_start_date';
	$fields['tblpur_contracts_end_date'] = 'tblpur_contracts_end_date';
	$fields['tblpur_contracts_signed_date'] = 'tblpur_contracts_signed_date';

	// tblpur_debit_notes
	$fields['tblpur_debit_notes_date'] = 'tblpur_debit_notes_date';

	// tblpur_debits
	$fields['tblpur_debits_date'] = 'tblpur_debits_date';

	// tblpur_debits_refunds
	$fields['tblpur_debits_refunds_refunded_on'] = 'tblpur_debits_refunds_refunded_on';

	// tblpur_estimates
	$fields['tblpur_estimates_date'] = 'tblpur_estimates_date';
	$fields['tblpur_estimates_expirydate'] = 'tblpur_estimates_expirydate';


	// tblpur_invoices
	$fields['tblpur_invoices_invoice_date'] = 'tblpur_invoices_invoice_date';
	$fields['tblpur_invoices_transaction_date'] = 'tblpur_invoices_transaction_date';
	$fields['tblpur_invoices_date_add'] = 'tblpur_invoices_date_add';

	// tblpur_invoice_payment
	$fields['tblpur_invoice_payment_date'] = 'tblpur_invoice_payment_date';

	// tblpur_orders
	$fields['tblpur_orders_order_date'] = 'tblpur_orders_order_date';
	$fields['tblpur_orders_delivery_date'] = 'tblpur_orders_delivery_date';

	// tblpur_request
	$fields['tblpur_request_request_date'] = 'tblpur_request_request_date';

	/*-----------Omnisalse module----------*/
	// tblcart
	$fields['tblcart_duedate'] = 'tblcart_duedate';
	// tblomni_trade_discount
	$fields['tblomni_trade_discount_start_time'] = 'tblomni_trade_discount_start_time';
	$fields['tblomni_trade_discount_end_time'] = 'tblomni_trade_discount_end_time';
	// tblomni_log_sync_woo
	$fields['tblomni_log_sync_woo_date_on_sale_from'] = 'tblomni_log_sync_woo_date_on_sale_from';
	$fields['tblomni_log_sync_woo_date_on_sale_to'] = 'tblomni_log_sync_woo_date_on_sale_to';




	return $fields;	
}

/**
 * rb datetime field
 * @return [type] 
 */
function rb_datetime_field()
{
	// operator: between 
	//input type: input datetime (two input)
	$fields = [];

	//tblclients
	$fields['tblclients_datecreated']=['tblclients_datecreated'];

	//tblcontact
	$fields['tblcontacts_datecreated'] = 'tblcontacts_datecreated';

	//tblinvoices
	$fields['tblinvoices_datesend'] = 'tblinvoices_datesend';
	$fields['tblinvoices_datecreated'] = 'tblinvoices_datecreated';

	//tblestimates
	$fields['tblestimates_datesend'] = 'tblestimates_datesend';
	$fields['tblestimates_datecreated'] = 'tblestimates_datecreated';
	$fields['tblestimates_invoiced_date'] = 'tblestimates_invoiced_date';

	//tblcreditnotes
	$fields['tblcreditnotes_datecreated'] = 'tblcreditnotes_datecreated';


	//tbltickets
	$fields['tbltickets_date'] = 'tbltickets_date';

	//subscriptions
	$fields['tblsubscriptions_date'] = 'tblsubscriptions_date';

	//tblcontracts
	$fields['tblcontracts_dateadded'] = 'tblcontracts_dateadded';
	$fields['tblcontracts_acceptance_date'] = 'tblcontracts_acceptance_date';

	//tbltasks
	$fields['tbltasks_dateadded'] = 'tbltasks_dateadded';
	$fields['tbltasks_datefinished'] = 'tbltasks_datefinished';

	//tbltask_comments
	$fields['tbltask_comments_dateadded'] = 'tbltask_comments_dateadded';

	//tblreminders
	$fields['tblreminders_date'] = 'tblreminders_date';

	//tblproposals
	$fields['tblproposals_datecreated'] = 'tblproposals_datecreated';
	$fields['tblproposals_date_converted'] = 'tblproposals_date_converted';
	$fields['tblproposals_acceptance_date'] = 'tblproposals_acceptance_date';

	//tblexpenses
	$fields['tblexpenses_dateadded'] = 'tblexpenses_dateadded';

	//tblleads
	$fields['tblleads_lastcontact'] = 'tblleads_lastcontact';
	$fields['tblleads_last_status_change'] = 'tblleads_last_status_change';
	$fields['tblleads_date_converted'] = 'tblleads_date_converted';

	//tblprojects
	$fields['tblprojects_date_finished'] = 'tblprojects_date_finished';

	//tblprojectdiscussions
	$fields['tblprojectdiscussions_datecreated'] = 'tblprojectdiscussions_datecreated';
	$fields['tblprojectdiscussions_last_activity'] = 'tblprojectdiscussions_last_activity';

	//tblprojectdiscussioncomments
	$fields['tblprojectdiscussioncomments_created'] = 'tblprojectdiscussioncomments_created';
	$fields['tblprojectdiscussioncomments_modified'] = 'tblprojectdiscussioncomments_modified';

	//tblstaff
	$fields['tblstaff_datecreated'] = 'tblstaff_datecreated';
	$fields['tblstaff_last_login'] = 'tblstaff_last_login';
	$fields['tblstaff_last_activity'] = 'tblstaff_last_activity';
	$fields['tblstaff_last_password_change'] = 'tblstaff_last_password_change';
	$fields['tblstaff_new_pass_key_requested'] = 'tblstaff_new_pass_key_requested';

	//tblproposal_comments
	$fields['tblproposal_comments_dateadded'] = 'tblproposal_comments_dateadded';

	//tblnotes
	$fields['tblnotes_date_contacted'] = 'tblnotes_date_contacted';
	$fields['tblnotes_dateadded'] = 'tblnotes_dateadded';

	//tblinvoicepaymentrecords
	$fields['tblinvoicepaymentrecords_daterecorded'] = 'tblinvoicepaymentrecords_daterecorded';

	//tblcredits
	$fields['tblcredits_date_applied'] = 'tblcredits_date_applied';

	//tblcreditnote_refunds
	$fields['tblcreditnote_refunds_created_at'] = 'tblcreditnote_refunds_created_at';

	//tblcontract_comments
	$fields['tblcontract_comments_dateadded'] = 'tblcontract_comments_dateadded';

	//tblcontract_renewals
	$fields['tblcontract_renewals_date_renewed'] = 'tblcontract_renewals_date_renewed';

	
	/*--------------warehouse module------------*/

	//tblinternal_delivery_note
	$fields['tblinternal_delivery_note_datecreated'] = 'tblinternal_delivery_note_datecreated';


	// tblpur_comments
	$fields['tblpur_comments_dateadded'] = 'tblpur_comments_dateadded';
	// tblpur_contacts
	
	$fields['tblpur_contacts_datecreated'] = 'tblpur_contacts_datecreated';
	$fields['tblpur_contacts_last_login'] = 'tblpur_contacts_last_login';
	$fields['tblpur_contacts_last_password_change'] = 'tblpur_contacts_last_password_change';

	// tblpur_debit_notes
	$fields['tblpur_debit_notes_datecreated'] = 'tblpur_debit_notes_datecreated';

	// tblpur_debits
	$fields['tblpur_debits_date_applied'] = 'tblpur_debits_date_applied';

	// tblpur_debits_refunds
	$fields['tblpur_debits_refunds_created_at'] = 'tblpur_debits_refunds_created_at';

	// tblpur_estimates
	$fields['tblpur_estimates_datesend'] = 'tblpur_estimates_datesend';
	$fields['tblpur_estimates_datecreated'] = 'tblpur_estimates_datecreated';
	$fields['tblpur_estimates_invoiced_date'] = 'tblpur_estimates_invoiced_date';
	$fields['tblpur_estimates_acceptance_date'] = 'tblpur_estimates_acceptance_date';

	
	// tblpur_invoice_payment
	$fields['tblpur_invoice_payment_daterecorded'] = 'tblpur_invoice_payment_daterecorded';

	// tblpur_orders
	$fields['tblpur_orders_datecreated'] = 'tblpur_orders_datecreated';

	// tblpur_vendor
	$fields['tblpur_vendor_datecreated'] = 'tblpur_vendor_datecreated';

	// tblpur_vendor_admin
	$fields['tblpur_vendor_admin_date_assigned'] = 'tblpur_vendor_admin_date_assigned';

	/*--------Omnisalse module-------*/
	// tblcart
	$fields['tblcart_datecreator'] = 'tblcart_datecreator';
	 // tblomni_log_sync_woo
	$fields['tblomni_log_sync_woo_date_sync'] = 'tblomni_log_sync_woo_date_sync';
	// tblomni_shift
	$fields['tblomni_shift_created_at'] = 'tblomni_shift_created_at';
	// tblomni_shift_history
	$fields['tblomni_shift_history_created_at'] = 'tblomni_shift_history_created_at';
	// tblomni_cart_payment
	$fields['tblomni_cart_payment_datecreator'] = 'tblomni_cart_payment_datecreator';
	// tblomni_pre_order_product_setting
	$fields['tblomni_pre_order_product_setting_datecreator'] = 'tblomni_pre_order_product_setting_datecreator';


	return $fields;
}

/**
 * rb primary foreign key get data
 * @param  [type] $data 
 * @return [type]       [description]
 */
function rb_primary_foreign_key_get_data($data)
{	
	$CI             = &get_instance();
	$result=[];
	if(is_array($data['field_name'])){
		$result = $data['field_name'];

	}else{
		$label = new_explode('#', $data['label']);
		$label1 = $label[0];
		if(count($label) > 1){
			$label2 = $label[1];
		}

		$table_data = $CI->db->get(db_prefix().$data['table_name'])->result_array();

		if(count($table_data) > 0){
			foreach ($table_data as $key => $table_value) {
				$label1_value = '';
				$label1_value = isset($table_value[$label1]) ? $table_value[$label1] : '';
				if(isset($label2)){
					$label1_value .= isset($table_value[$label2]) ? ' '.$table_value[$label2] : '';
				}

				$label_value = 
				$result[] = [
					'value' => $table_value[$data['field_name']],
					'label' => $label1_value,
				];
			}	
		}

	}

	return $result;

}

/**
 * rb report detail format value
 * @param  [type] $value        
 * @param  [type] $table_column 
 * @return [type]               
 */
function rb_report_detail_format_value($value, $table_column)
{
	
	$rb_primary_foreign_key_field = rb_primary_foreign_key_field();
	$rb_number_field = rb_number_field();
	$rb_date_field = rb_date_field();
	$rb_datetime_field = rb_datetime_field();

	if(isset($rb_primary_foreign_key_field[new_str_replace('.', '_', $table_column)]) ){

	}elseif(isset($rb_number_field[new_str_replace('.', '_', $table_column)])){
		$value = app_format_money((float)$value, '');
	}elseif(isset($rb_date_field[new_str_replace('.', '_', $table_column)])){
		$value = _d($value);

	}elseif(isset($rb_datetime_field[new_str_replace('.', '_', $table_column)])){
		$value = _dt($value);

	}

	return $value;

}

/**
 * rb cell formatting color
 * @param  [type] $table_column     
 * @param  [type] $cell_formattings 
 * @return [type]                   
 */
function rb_cell_formatting_color($table_column, $cell_formattings, $value)
{
	$color = '#000000';
	if(isset($cell_formattings[$table_column]) && $value){

		switch($cell_formattings[$table_column]['filter_type']) {
			case 'equal':
			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {

				if((float)$get_filter_value['filter_value_1'] == (float)$value){
					$color = $get_filter_value['color_hex'];
				}
			}
			break;

			case 'greater_than':
			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {
				if((float)$value > (float)$get_filter_value['filter_value_1']){
					$color = $get_filter_value['color_hex'];
				}
			}
			break;

			case 'less_than':
			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {
				if((float)$value < (float)$get_filter_value['filter_value_1']){
					$color = $get_filter_value['color_hex'];
				}
			}
			break;

			case 'greater_than_or_equal':
			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {
				if((float)$value >= (float)$get_filter_value['filter_value_1']){
					$color = $get_filter_value['color_hex'];
				}
			}
			break;

			case 'less_than_or_equal':
			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {
				if((float)$value <= (float)$get_filter_value['filter_value_1']){
					$color = $get_filter_value['color_hex'];
				}
			}
			break;

			case 'between':

			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {
				if (preg_match("/^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])$/", to_sql_date($get_filter_value['filter_value_1']) ) == 1) {
				
					if(strtotime($value) >= strtotime(to_sql_date($get_filter_value['filter_value_1'])) && strtotime($value) <= strtotime(to_sql_date($get_filter_value['filter_value_2'])) ){
						$color = $get_filter_value['color_hex'];
					}

				}elseif(preg_match("/\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/i", to_sql_date($get_filter_value['filter_value_1'], true) ) == 1) {

					if(strtotime($value) >= strtotime(to_sql_date($get_filter_value['filter_value_1'], true)) && strtotime($value) <= strtotime(to_sql_date($get_filter_value['filter_value_2'], true)) ){
						$color = $get_filter_value['color_hex'];
					}
				}else{

					if((float)$value >= (float)$get_filter_value['filter_value_1'] && (float)$value <= (float)$get_filter_value['filter_value_2']){
						$color = $get_filter_value['color_hex'];
					}
				}
			}
			
			break;

			case 'like':
			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {
				if(strpos($get_filter_value['filter_value_1'], $value)){
					$color = $get_filter_value['color_hex'];
				}
			}
			break;

			case 'NOT_like':
			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {
				if(!strpos($get_filter_value['filter_value_1'], $value)){
					$color = $get_filter_value['color_hex'];
				}
			}
			break;

			case 'not_equal':
			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {
				if($value != $get_filter_value['filter_value_1']){
					$color = $get_filter_value['color_hex'];
				}
			}
			break;

			case 'begin_with':
			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {
				if(preg_match('/^'.$get_filter_value['filter_value_1'].'/', $value) == 1){
					$color = $get_filter_value['color_hex'];
				}
			}
			break;

			case 'end_with':
			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {
				if(preg_match('/'.$get_filter_value['filter_value_1'].'^/', $value) == 1){
					$color = $get_filter_value['color_hex'];
				}
			}
			break;

			case 'in':
			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {
				if(strpos($get_filter_value['filter_value_1'], $value)){
					$color = $get_filter_value['color_hex'];
				}
			}
			break;
			case 'not_in':
			foreach ($cell_formattings[$table_column]['filter_values'] as $get_filter_value) {
				if(!strpos($get_filter_value['filter_value_1'], $value)){
					$color = $get_filter_value['color_hex'];
				}
			}

			break;
		}
	}

	return 'color:'.$color.';';
}

/**
 * new html entity decode
 * @param  [type] $str 
 * @return [type]      
 */
if (!function_exists('new_html_entity_decode')) {
	function new_html_entity_decode($str){
		return html_entity_decode($str ?? '');
	}
}

if (!function_exists('new_strlen')) {
	
	function new_strlen($str){
		return strlen($str ?? '');
	}
}

if (!function_exists('new_str_replace')) {
	
	function new_str_replace($search, $replace, $subject){
		return str_replace($search, $replace, $subject ?? '');
	}
}

if (!function_exists('new_explode')) {

	function new_explode($delimiter, $string){
		return explode($delimiter, $string ?? '');
	}
}