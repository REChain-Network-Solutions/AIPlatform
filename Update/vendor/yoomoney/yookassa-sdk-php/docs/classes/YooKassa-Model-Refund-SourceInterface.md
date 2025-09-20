# [YooKassa API SDK](../home.md)

# Interface: SourceInterface
### Namespace: [\YooKassa\Model\Refund](../namespaces/yookassa-model-refund.md)
---
**Summary:**

Interface SourceInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAccountId()](../classes/YooKassa-Model-Refund-SourceInterface.md#method_getAccountId) |  | Возвращает id магазина с которого будут списаны средства. |
| public | [getAmount()](../classes/YooKassa-Model-Refund-SourceInterface.md#method_getAmount) |  | Возвращает сумму оплаты. |
| public | [getPlatformFeeAmount()](../classes/YooKassa-Model-Refund-SourceInterface.md#method_getPlatformFeeAmount) |  | Возвращает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу. |
| public | [hasAmount()](../classes/YooKassa-Model-Refund-SourceInterface.md#method_hasAmount) |  | Проверяет, была ли установлена сумма оплаты. |
| public | [hasPlatformFeeAmount()](../classes/YooKassa-Model-Refund-SourceInterface.md#method_hasPlatformFeeAmount) |  | Проверяет, была ли установлена комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу. |
| public | [setAccountId()](../classes/YooKassa-Model-Refund-SourceInterface.md#method_setAccountId) |  | Устанавливает id магазина-получателя средств. |
| public | [setAmount()](../classes/YooKassa-Model-Refund-SourceInterface.md#method_setAmount) |  | Устанавливает сумму оплаты. |
| public | [setPlatformFeeAmount()](../classes/YooKassa-Model-Refund-SourceInterface.md#method_setPlatformFeeAmount) |  | Устанавливает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу. |

---
### Details
* File: [lib/Model/Refund/SourceInterface.php](../../lib/Model/Refund/SourceInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |
| property |  | Сумма возврата |
| property |  | Комиссия, которую вы удержали при оплате, и хотите вернуть |
| property |  | Комиссия, которую вы удержали при оплате, и хотите вернуть |
| property |  | Идентификатор магазина, для которого вы хотите провести возврат |
| property |  | Идентификатор магазина, для которого вы хотите провести возврат |

---
## Methods
<a name="method_setAccountId" class="anchor"></a>
#### public setAccountId() : \YooKassa\Model\Refund\SourceInterface

```php
public setAccountId(string|null $account_id) : \YooKassa\Model\Refund\SourceInterface
```

**Summary**

Устанавливает id магазина-получателя средств.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\SourceInterface](../classes/YooKassa-Model-Refund-SourceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | account_id  | Id магазина с которого будут списаны средства |

**Returns:** \YooKassa\Model\Refund\SourceInterface - 


<a name="method_getAccountId" class="anchor"></a>
#### public getAccountId() : string|null

```php
public getAccountId() : string|null
```

**Summary**

Возвращает id магазина с которого будут списаны средства.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\SourceInterface](../classes/YooKassa-Model-Refund-SourceInterface.md)

**Returns:** string|null - Id магазина с которого будут списаны средства


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\SourceInterface](../classes/YooKassa-Model-Refund-SourceInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма оплаты


<a name="method_hasAmount" class="anchor"></a>
#### public hasAmount() : bool

```php
public hasAmount() : bool
```

**Summary**

Проверяет, была ли установлена сумма оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\SourceInterface](../classes/YooKassa-Model-Refund-SourceInterface.md)

**Returns:** bool - True если сумма оплаты была установлена, false если нет


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : \YooKassa\Model\Refund\SourceInterface

```php
public setAmount(\YooKassa\Model\AmountInterface|array|null $amount = null) : \YooKassa\Model\Refund\SourceInterface
```

**Summary**

Устанавливает сумму оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\SourceInterface](../classes/YooKassa-Model-Refund-SourceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | amount  | Сумма оплаты |

**Returns:** \YooKassa\Model\Refund\SourceInterface - 


<a name="method_getPlatformFeeAmount" class="anchor"></a>
#### public getPlatformFeeAmount() : \YooKassa\Model\AmountInterface|null

```php
public getPlatformFeeAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\SourceInterface](../classes/YooKassa-Model-Refund-SourceInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма комиссии


<a name="method_hasPlatformFeeAmount" class="anchor"></a>
#### public hasPlatformFeeAmount() : bool

```php
public hasPlatformFeeAmount() : bool
```

**Summary**

Проверяет, была ли установлена комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\SourceInterface](../classes/YooKassa-Model-Refund-SourceInterface.md)

**Returns:** bool - True если комиссия была установлена, false если нет


<a name="method_setPlatformFeeAmount" class="anchor"></a>
#### public setPlatformFeeAmount() : \YooKassa\Model\Refund\SourceInterface

```php
public setPlatformFeeAmount(\YooKassa\Model\AmountInterface|array|null $platform_fee_amount) : \YooKassa\Model\Refund\SourceInterface
```

**Summary**

Устанавливает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\SourceInterface](../classes/YooKassa-Model-Refund-SourceInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | platform_fee_amount  | Сумма комиссии |

**Returns:** \YooKassa\Model\Refund\SourceInterface - 




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