# [YooKassa API SDK](../home.md)

# Interface: TransferDataInterface
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Interface TransferDataInterface.

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
| public | [getAccountId()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_getAccountId) |  | Возвращает account_id. |
| public | [getAmount()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_getAmount) |  | Возвращает amount. |
| public | [getDescription()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_getDescription) |  | Возвращает description. |
| public | [getMetadata()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_getMetadata) |  | Возвращает metadata. |
| public | [getPlatformFeeAmount()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_getPlatformFeeAmount) |  | Возвращает platform_fee_amount. |
| public | [hasAccountId()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_hasAccountId) |  |  |
| public | [hasAmount()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_hasAmount) |  |  |
| public | [hasDescription()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_hasDescription) |  |  |
| public | [hasMetadata()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_hasMetadata) |  |  |
| public | [hasPlatformFeeAmount()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_hasPlatformFeeAmount) |  |  |
| public | [setAccountId()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_setAccountId) |  | Устанавливает account_id. |
| public | [setAmount()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_setAmount) |  | Устанавливает amount. |
| public | [setDescription()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_setDescription) |  | Устанавливает description. |
| public | [setMetadata()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_setMetadata) |  | Устанавливает metadata. |
| public | [setPlatformFeeAmount()](../classes/YooKassa-Request-Payments-TransferDataInterface.md#method_setPlatformFeeAmount) |  | Устанавливает platform_fee_amount. |

---
### Details
* File: [lib/Request/Payments/TransferDataInterface.php](../../lib/Request/Payments/TransferDataInterface.php)
* Package: \YooKassa\Request
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
| property |  | Описание транзакции, которое продавец увидит в личном кабинете ЮKassa. (например: «Заказ маркетплейса №72») |
| property |  | Любые дополнительные данные, которые нужны вам для работы с платежами (например, ваш внутренний идентификатор заказа) |

---
## Methods
<a name="method_getAccountId" class="anchor"></a>
#### public getAccountId() : string|null

```php
public getAccountId() : string|null
```

**Summary**

Возвращает account_id.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

**Returns:** string|null - 


<a name="method_setAccountId" class="anchor"></a>
#### public setAccountId() : \YooKassa\Request\Payments\TransferData

```php
public setAccountId(string|null $value = null) : \YooKassa\Request\Payments\TransferData
```

**Summary**

Устанавливает account_id.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  |  |

**Returns:** \YooKassa\Request\Payments\TransferData - 


<a name="method_hasAccountId" class="anchor"></a>
#### public hasAccountId() : bool

```php
public hasAccountId() : bool
```

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

**Returns:** bool - 


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает amount.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - 


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : \YooKassa\Request\Payments\TransferData

```php
public setAmount(\YooKassa\Model\AmountInterface|array|null $value = null) : \YooKassa\Request\Payments\TransferData
```

**Summary**

Устанавливает amount.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | value  |  |

**Returns:** \YooKassa\Request\Payments\TransferData - 


<a name="method_hasAmount" class="anchor"></a>
#### public hasAmount() : bool

```php
public hasAmount() : bool
```

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

**Returns:** bool - 


<a name="method_getPlatformFeeAmount" class="anchor"></a>
#### public getPlatformFeeAmount() : \YooKassa\Model\AmountInterface|null

```php
public getPlatformFeeAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает platform_fee_amount.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - 


<a name="method_setPlatformFeeAmount" class="anchor"></a>
#### public setPlatformFeeAmount() : \YooKassa\Request\Payments\TransferData

```php
public setPlatformFeeAmount(\YooKassa\Model\AmountInterface|array|null $value = null) : \YooKassa\Request\Payments\TransferData
```

**Summary**

Устанавливает platform_fee_amount.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | value  |  |

**Returns:** \YooKassa\Request\Payments\TransferData - 


<a name="method_hasPlatformFeeAmount" class="anchor"></a>
#### public hasPlatformFeeAmount() : bool

```php
public hasPlatformFeeAmount() : bool
```

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

**Returns:** bool - 


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает description.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

**Returns:** string|null - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : \YooKassa\Request\Payments\TransferData

```php
public setDescription(string|null $value = null) : \YooKassa\Request\Payments\TransferData
```

**Summary**

Устанавливает description.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Описание транзакции (не более 128 символов), которое продавец увидит в личном кабинете ЮKassa. Например: «Заказ маркетплейса №72». |

**Returns:** \YooKassa\Request\Payments\TransferData - 


<a name="method_hasDescription" class="anchor"></a>
#### public hasDescription() : bool

```php
public hasDescription() : bool
```

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

**Returns:** bool - 


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает metadata.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

**Returns:** \YooKassa\Model\Metadata|null - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : \YooKassa\Request\Payments\TransferData

```php
public setMetadata(string|array|null $value = null) : \YooKassa\Request\Payments\TransferData
```

**Summary**

Устанавливает metadata.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR array OR null</code> | value  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8. |

**Returns:** \YooKassa\Request\Payments\TransferData - 


<a name="method_hasMetadata" class="anchor"></a>
#### public hasMetadata() : bool

```php
public hasMetadata() : bool
```

**Details:**
* Inherited From: [\YooKassa\Request\Payments\TransferDataInterface](../classes/YooKassa-Request-Payments-TransferDataInterface.md)

**Returns:** bool - 




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