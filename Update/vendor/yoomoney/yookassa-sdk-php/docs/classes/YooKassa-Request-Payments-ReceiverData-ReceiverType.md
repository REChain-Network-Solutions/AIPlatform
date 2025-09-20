# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\ReceiverData\ReceiverType
### Namespace: [\YooKassa\Request\Payments\ReceiverData](../namespaces/yookassa-request-payments-receiverdata.md)
---
**Summary:**

Класс, представляющий модель ReceiverType.

**Description:**

Код получателя оплаты.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [MOBILE_BALANCE](../classes/YooKassa-Request-Payments-ReceiverData-ReceiverType.md#constant_MOBILE_BALANCE) |  | Пополнение баланса телефона |
| public | [DIGITAL_WALLET](../classes/YooKassa-Request-Payments-ReceiverData-ReceiverType.md#constant_DIGITAL_WALLET) |  | Пополнение электронного кошелька |
| public | [BANK_ACCOUNT](../classes/YooKassa-Request-Payments-ReceiverData-ReceiverType.md#constant_BANK_ACCOUNT) |  | Пополнение банковского счета, открытого в вашей системе |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Request-Payments-ReceiverData-ReceiverType.md#property_validValues) |  | Возвращает список доступных значений |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Request/Payments/ReceiverData/ReceiverType.php](../../lib/Request/Payments/ReceiverData/ReceiverType.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Request\Payments\ReceiverData\ReceiverType

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_MOBILE_BALANCE" class="anchor"></a>
###### MOBILE_BALANCE
Пополнение баланса телефона

```php
MOBILE_BALANCE = 'mobile_balance'
```


<a name="constant_DIGITAL_WALLET" class="anchor"></a>
###### DIGITAL_WALLET
Пополнение электронного кошелька

```php
DIGITAL_WALLET = 'digital_wallet'
```


<a name="constant_BANK_ACCOUNT" class="anchor"></a>
###### BANK_ACCOUNT
Пополнение банковского счета, открытого в вашей системе

```php
BANK_ACCOUNT = 'bank_account'
```



---
## Properties
<a name="property_validValues"></a>
#### protected $validValues : array
---
**Summary**

Возвращает список доступных значений

**Type:** <a href="../array"><abbr title="array">array</abbr></a>
Массив принимаемых enum&#039;ом значений
**Details:**


##### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| return |  |  |


---
## Methods
<a name="method_getEnabledValues" class="anchor"></a>
#### public getEnabledValues() : string[]

```php
Static public getEnabledValues() : string[]
```

**Summary**

Возвращает значения в enum'е значения которых разрешены.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

**Returns:** string[] - Массив разрешённых значений


<a name="method_getValidValues" class="anchor"></a>
#### public getValidValues() : array

```php
Static public getValidValues() : array
```

**Summary**

Возвращает все значения в enum'e.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

**Returns:** array - Массив значений в перечислении


<a name="method_valueExists" class="anchor"></a>
#### public valueExists() : bool

```php
Static public valueExists(mixed $value) : bool
```

**Summary**

Проверяет наличие значения в enum'e.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | value  | Проверяемое значение |

**Returns:** bool - True если значение имеется, false если нет



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