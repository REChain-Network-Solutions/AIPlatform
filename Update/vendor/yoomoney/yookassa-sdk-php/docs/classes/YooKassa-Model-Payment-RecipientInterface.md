# [YooKassa API SDK](../home.md)

# Interface: RecipientInterface
### Namespace: [\YooKassa\Model\Payment](../namespaces/yookassa-model-payment.md)
---
**Summary:**

Интерфейс получателя платежа.

**Description:**

Получатель платежа нужен, если вы разделяете потоки платежей в рамках одного аккаунта или создаете платеж в адрес
другого аккаунта.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAccountId()](../classes/YooKassa-Model-Payment-RecipientInterface.md#method_getAccountId) |  | Возвращает идентификатор магазина. |
| public | [getGatewayId()](../classes/YooKassa-Model-Payment-RecipientInterface.md#method_getGatewayId) |  | Возвращает идентификатор шлюза. |

---
### Details
* File: [lib/Model/Payment/RecipientInterface.php](../../lib/Model/Payment/RecipientInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Идентификатор магазина |
| property |  | Идентификатор магазина |
| property |  | Идентификатор шлюза |
| property |  | Идентификатор шлюза |

---
## Methods
<a name="method_getAccountId" class="anchor"></a>
#### public getAccountId() : string|null

```php
public getAccountId() : string|null
```

**Summary**

Возвращает идентификатор магазина.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\RecipientInterface](../classes/YooKassa-Model-Payment-RecipientInterface.md)

**Returns:** string|null - Идентификатор магазина


<a name="method_getGatewayId" class="anchor"></a>
#### public getGatewayId() : string|null

```php
public getGatewayId() : string|null
```

**Summary**

Возвращает идентификатор шлюза.

**Description**

Идентификатор шлюза используется для разделения потоков платежей в рамках одного аккаунта.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\RecipientInterface](../classes/YooKassa-Model-Payment-RecipientInterface.md)

**Returns:** string|null - Идентификатор шлюза




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