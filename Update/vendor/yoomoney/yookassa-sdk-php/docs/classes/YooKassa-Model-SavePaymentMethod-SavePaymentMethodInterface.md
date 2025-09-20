# [YooKassa API SDK](../home.md)

# Interface: SavePaymentMethodInterface
### Namespace: [\YooKassa\Model\SavePaymentMethod](../namespaces/yookassa-model-savepaymentmethod.md)
---
**Summary:**

Interface SavePaymentMethodInterface.

**Description:**

Сохраненный способ оплаты.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getCard()](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md#method_getCard) |  | Возвращает card. |
| public | [getConfirmation()](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md#method_getConfirmation) |  | Возвращает выбранный сценарий подтверждения привязки. |
| public | [getHolder()](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md#method_getHolder) |  | Возвращает данные магазина, для которого сохраняется способ оплаты. |
| public | [getId()](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md#method_getId) |  | Возвращает идентификатор сохраненного способа оплаты. |
| public | [getSaved()](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md#method_getSaved) |  | Возвращает признак сохранения способа оплаты для %[автоплатежей](/developers/payment-acceptance/scenario-extensions/recurring-payments/pay-with-saved).  Возможные значения:   * ~`true` — способ оплаты сохранен для автоплатежей и выплат; * ~`false` — способ оплаты не сохранен. |
| public | [getStatus()](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md#method_getStatus) |  | Возвращает статус проверки и сохранения способа оплаты. |
| public | [getTitle()](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md#method_getTitle) |  | Возвращает название способа оплаты. |
| public | [getType()](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md#method_getType) |  | Возвращает код способа оплаты. Возможное значение: ~`bank_card` — банковская карта. |

---
### Details
* File: [lib/Model/SavePaymentMethod/SavePaymentMethodInterface.php](../../lib/Model/SavePaymentMethod/SavePaymentMethodInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Код способа оплаты.  Возможное значение: ~`bank_card` — банковская карта. |
| property |  | Идентификатор сохраненного способа оплаты. |
| property |  | Признак сохранения способа оплаты для [автоплатежей](/developers/payment-acceptance/scenario-extensions/recurring-payments/pay-with-saved).  Возможные значения:   * ~`true` — способ оплаты сохранен для автоплатежей и выплат; * ~`false` — способ оплаты не сохранен. |
| property |  | Статус проверки и сохранения способа оплаты. |
| property |  | Данные магазина, для которого сохраняется способ оплаты. |
| property |  | Название способа оплаты. |
| property |  | Выбранный сценарий подтверждения привязки. Присутствует, когда привязка ожидает подтверждения от пользователя. |

---
## Methods
<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает код способа оплаты. Возможное значение: ~`bank_card` — банковская карта.

**Details:**
* Inherited From: [\YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md)

**Returns:** string|null - 


<a name="method_getCard" class="anchor"></a>
#### public getCard() : \YooKassa\Model\Payment\PaymentMethod\BankCard|null

```php
public getCard() : \YooKassa\Model\Payment\PaymentMethod\BankCard|null
```

**Summary**

Возвращает card.

**Details:**
* Inherited From: [\YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md)

**Returns:** \YooKassa\Model\Payment\PaymentMethod\BankCard|null - 


<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает идентификатор сохраненного способа оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md)

**Returns:** string|null - 


<a name="method_getSaved" class="anchor"></a>
#### public getSaved() : bool|null

```php
public getSaved() : bool|null
```

**Summary**

Возвращает признак сохранения способа оплаты для %[автоплатежей](/developers/payment-acceptance/scenario-extensions/recurring-payments/pay-with-saved).  Возможные значения:   * ~`true` — способ оплаты сохранен для автоплатежей и выплат; * ~`false` — способ оплаты не сохранен.

**Details:**
* Inherited From: [\YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md)

**Returns:** bool|null - 


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает статус проверки и сохранения способа оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md)

**Returns:** string|null - 


<a name="method_getHolder" class="anchor"></a>
#### public getHolder() : \YooKassa\Model\SavePaymentMethod\SavePaymentMethodHolder|null

```php
public getHolder() : \YooKassa\Model\SavePaymentMethod\SavePaymentMethodHolder|null
```

**Summary**

Возвращает данные магазина, для которого сохраняется способ оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md)

**Returns:** \YooKassa\Model\SavePaymentMethod\SavePaymentMethodHolder|null - 


<a name="method_getTitle" class="anchor"></a>
#### public getTitle() : string|null

```php
public getTitle() : string|null
```

**Summary**

Возвращает название способа оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md)

**Returns:** string|null - 


<a name="method_getConfirmation" class="anchor"></a>
#### public getConfirmation() : \YooKassa\Model\SavePaymentMethod\Confirmation\AbstractConfirmation|null

```php
public getConfirmation() : \YooKassa\Model\SavePaymentMethod\Confirmation\AbstractConfirmation|null
```

**Summary**

Возвращает выбранный сценарий подтверждения привязки.

**Details:**
* Inherited From: [\YooKassa\Model\SavePaymentMethod\SavePaymentMethodInterface](../classes/YooKassa-Model-SavePaymentMethod-SavePaymentMethodInterface.md)

**Returns:** \YooKassa\Model\SavePaymentMethod\Confirmation\AbstractConfirmation|null - 




---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney