# [YooKassa API SDK](../home.md)

# Interface: TransferInterface
### Namespace: [\YooKassa\Model\Payment](../namespaces/yookassa-model-payment.md)
---
**Summary:**

Interface TransferInterface.

**Description:**

Данные о распределении денег — сколько и в какой магазин нужно перевести.
Присутствует, если вы используете [Сплитование платежей](/developers/solutions-for-platforms/split-payments/basics).

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAccountId()](../classes/YooKassa-Model-Payment-TransferInterface.md#method_getAccountId) |  | Возвращает идентификатор магазина-получателя средств. |
| public | [getAmount()](../classes/YooKassa-Model-Payment-TransferInterface.md#method_getAmount) |  | Возвращает сумму оплаты. |
| public | [getDescription()](../classes/YooKassa-Model-Payment-TransferInterface.md#method_getDescription) |  | Возвращает описание транзакции. |
| public | [getMetadata()](../classes/YooKassa-Model-Payment-TransferInterface.md#method_getMetadata) |  | Возвращает метаданные. |
| public | [getPlatformFeeAmount()](../classes/YooKassa-Model-Payment-TransferInterface.md#method_getPlatformFeeAmount) |  | Возвращает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу. |
| public | [getStatus()](../classes/YooKassa-Model-Payment-TransferInterface.md#method_getStatus) |  | Возвращает статус операции распределения средств конечному получателю. |
| public | [setAccountId()](../classes/YooKassa-Model-Payment-TransferInterface.md#method_setAccountId) |  | Устанавливает идентификатор магазина-получателя средств. |
| public | [setAmount()](../classes/YooKassa-Model-Payment-TransferInterface.md#method_setAmount) |  | Устанавливает сумму оплаты. |
| public | [setDescription()](../classes/YooKassa-Model-Payment-TransferInterface.md#method_setDescription) |  | Устанавливает описание транзакции. |
| public | [setMetadata()](../classes/YooKassa-Model-Payment-TransferInterface.md#method_setMetadata) |  | Устанавливает метаданные. |
| public | [setPlatformFeeAmount()](../classes/YooKassa-Model-Payment-TransferInterface.md#method_setPlatformFeeAmount) |  | Устанавливает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу. |
| public | [setStatus()](../classes/YooKassa-Model-Payment-TransferInterface.md#method_setStatus) |  | Устанавливает статус операции распределения средств конечному получателю. |

---
### Details
* File: [lib/Model/Payment/TransferInterface.php](../../lib/Model/Payment/TransferInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Идентификатор магазина, в пользу которого вы принимаете оплату |
| property |  | Идентификатор магазина, в пользу которого вы принимаете оплату |
| property |  | Сумма, которую необходимо перечислить магазину |
| property |  | Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу |
| property |  | Комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу |
| property |  | Статус распределения денег между магазинами. Возможные значения: `pending`, `waiting_for_capture`, `succeeded`, `canceled` |
| property |  | Описание транзакции, которое продавец увидит в личном кабинете ЮKassa. (например: «Заказ маркетплейса №72») |
| property |  | Любые дополнительные данные, которые нужны вам для работы с платежами (например, ваш внутренний идентификатор заказа) |
| property |  | Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать |
| property |  | Порядок перевода денег продавцам: ~`true` — перевести сразу, ~`false` — сначала захолдировать |
| property |  | Идентификатор продавца, подключенного к вашей платформе |
| property |  | Идентификатор продавца, подключенного к вашей платформе |

---
## Methods
<a name="method_setAccountId" class="anchor"></a>
#### public setAccountId() : self

```php
public setAccountId(string $value) : self
```

**Summary**

Устанавливает идентификатор магазина-получателя средств.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | Идентификатор магазина-получателя средств |

**Returns:** self - 


<a name="method_getAccountId" class="anchor"></a>
#### public getAccountId() : null|string

```php
public getAccountId() : null|string
```

**Summary**

Возвращает идентификатор магазина-получателя средств.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

**Returns:** null|string - Идентификатор магазина-получателя средств


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма оплаты


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : self

```php
public setAmount(\YooKassa\Model\AmountInterface|array|null $value) : self
```

**Summary**

Устанавливает сумму оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | value  | Сумма оплаты |

**Returns:** self - 


<a name="method_getPlatformFeeAmount" class="anchor"></a>
#### public getPlatformFeeAmount() : \YooKassa\Model\AmountInterface|null

```php
public getPlatformFeeAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма комиссии


<a name="method_setPlatformFeeAmount" class="anchor"></a>
#### public setPlatformFeeAmount() : self

```php
public setPlatformFeeAmount(\YooKassa\Model\AmountInterface|array|null $value) : self
```

**Summary**

Устанавливает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | value  | Сумма комиссии |

**Returns:** self - 


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : null|string

```php
public getStatus() : null|string
```

**Summary**

Возвращает статус операции распределения средств конечному получателю.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

**Returns:** null|string - Статус операции распределения средств конечному получателю


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : mixed

```php
public setStatus(?string $value) : mixed
```

**Summary**

Устанавливает статус операции распределения средств конечному получателю.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">?string</code> | value  |  |

**Returns:** mixed - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(null|string $value) : self
```

**Summary**

Устанавливает описание транзакции.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | value  | Описание транзакции |

**Returns:** self - 


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : null|string

```php
public getDescription() : null|string
```

**Summary**

Возвращает описание транзакции.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

**Returns:** null|string - Описание транзакции


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(null|array|\YooKassa\Model\Metadata $value) : self
```

**Summary**

Устанавливает метаданные.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Metadata</code> | value  | Метаданные |

**Returns:** self - 


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : null|\YooKassa\Model\Metadata

```php
public getMetadata() : null|\YooKassa\Model\Metadata
```

**Summary**

Возвращает метаданные.

**Details:**
* Inherited From: [\YooKassa\Model\Payment\TransferInterface](../classes/YooKassa-Model-Payment-TransferInterface.md)

**Returns:** null|\YooKassa\Model\Metadata - Метаданные




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