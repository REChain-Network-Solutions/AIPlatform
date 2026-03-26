<?php

return {
    'module': 'inspections',
    'capabilities': [
        {
            'key': 'inspections.help.explain_page',
            'label': 'Inspection: Explain this page',
            'risk': 'low',
            'requires': [],
            'handler': 'titanzero.intent.explain_page',
            'voice_phrases': [
                'what is this page',
                'explain this',
                'help me'
            ]
        },
        {
            'key': 'inspections.inspection-templates.items.destroy',
            'label': 'Inspection: Destroy',
            'risk': 'low',
            'requires': [],
            'handler': 'inspection-templates.items.destroy',
            'voice_phrases': [
                'destroy'
            ]
        },
        {
            'key': 'inspections.inspection-templates.items.store',
            'label': 'Inspection: Store',
            'risk': 'low',
            'requires': [],
            'handler': 'inspection-templates.items.store',
            'voice_phrases': [
                'store'
            ]
        },
        {
            'key': 'inspections.inspection.apply_quick_action',
            'label': 'Inspection: Apply Quick Action',
            'risk': 'low',
            'requires': [],
            'handler': 'inspection.apply_quick_action',
            'voice_phrases': [
                'apply quick action'
            ]
        },
        {
            'key': 'inspections.recurring_schedule.change_status',
            'label': 'Inspection: Change Status',
            'risk': 'low',
            'requires': [],
            'handler': 'recurring_schedule.change_status',
            'voice_phrases': [
                'change status'
            ]
        },
        {
            'key': 'inspections.recurring_schedule.export',
            'label': 'Inspection: Export',
            'risk': 'low',
            'requires': [],
            'handler': 'recurring_schedule.export',
            'voice_phrases': [
                'export'
            ]
        },
        {
            'key': 'inspections.recurring_schedule.recurring_schedule',
            'label': 'Inspection: Recurring Schedule',
            'risk': 'low',
            'requires': [],
            'handler': 'recurring_schedule.recurring_schedule',
            'voice_phrases': [
                'recurring schedule'
            ]
        },
        {
            'key': 'inspections.schedule-files.download',
            'label': 'Inspection: Download',
            'risk': 'low',
            'requires': [],
            'handler': 'schedule-files.download',
            'voice_phrases': [
                'download'
            ]
        },
        {
            'key': 'inspections.schedule-inspection.change-status',
            'label': 'Inspection: Change Status',
            'risk': 'low',
            'requires': [],
            'handler': 'schedule-inspection.change-status',
            'voice_phrases': [
                'change status'
            ]
        },
        {
            'key': 'inspections.schedule-inspection.refresh_count',
            'label': 'Inspection: Refresh Count',
            'risk': 'low',
            'requires': [],
            'handler': 'schedule-inspection.refresh_count',
            'voice_phrases': [
                'refresh count'
            ]
        },
        {
            'key': 'inspections.schedule-inspection.update_other_data',
            'label': 'Inspection: Update Other Data',
            'risk': 'low',
            'requires': [],
            'handler': 'schedule-inspection.update_other_data',
            'voice_phrases': [
                'update other data'
            ]
        },
        {
            'key': 'inspections.schedules.apply_quick_action',
            'label': 'Inspection: Apply Quick Action',
            'risk': 'low',
            'requires': [],
            'handler': 'schedules.apply_quick_action',
            'voice_phrases': [
                'apply quick action'
            ]
        },
        {
            'key': 'inspections.schedules.update_status',
            'label': 'Inspection: Update Status',
            'risk': 'low',
            'requires': [],
            'handler': 'schedules.update_status',
            'voice_phrases': [
                'update status'
            ]
        }
    ],
    'go_enabled': true,
    'zero_enabled': true
};
