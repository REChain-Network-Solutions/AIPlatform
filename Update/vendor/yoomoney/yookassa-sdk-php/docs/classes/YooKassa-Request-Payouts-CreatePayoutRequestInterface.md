# [YooKassa API SDK](../home.md)

# Interface: CreatePayoutRequestInterface
### Namespace: [\YooKassa\Request\Payouts](../namespaces/yookassa-request-payouts.md)
---
**Summary:**

Interface CreatePayoutRequestInterface.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getAmount()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_getAmount) |  | Возвращает сумму выплаты. |
| public | [getDeal()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_getDeal) |  | Возвращает сделку, в рамках которой нужно провести выплату. |
| public | [getDescription()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_getDescription) |  | Возвращает описание транзакции. |
| public | [getMetadata()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_getMetadata) |  | Возвращает данные оплаты установленные мерчантом |
| public | [getPaymentMethodId()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_getPaymentMethodId) |  | Возвращает идентификатор сохраненного способа оплаты. |
| public | [getPayoutDestinationData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_getPayoutDestinationData) |  | Возвращает данные платежного средства, на которое нужно сделать выплату. |
| public | [getPayoutToken()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_getPayoutToken) |  | Возвращает токенизированные данные для выплаты. |
| public | [getPersonalData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_getPersonalData) |  | Возвращает персональные данные получателя выплаты. |
| public | [getReceiptData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_getReceiptData) |  | Возвращает данные для формирования чека в сервисе Мой налог. |
| public | [getSelfEmployed()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_getSelfEmployed) |  | Возвращает данные самозанятого, который получит выплату. |
| public | [hasAmount()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_hasAmount) |  | Проверяет наличие суммы в создаваемой выплате. |
| public | [hasDeal()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_hasDeal) |  | Проверяет установлена ли сделка, в рамках которой нужно провести выплату. |
| public | [hasDescription()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_hasDescription) |  | Проверяет наличие описания транзакции в создаваемой выплате. |
| public | [hasMetadata()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_hasMetadata) |  | Проверяет, были ли установлены метаданные заказа. |
| public | [hasPaymentMethodId()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_hasPaymentMethodId) |  | Проверяет наличие идентификатора сохраненного способа оплаты. |
| public | [hasPayoutDestinationData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_hasPayoutDestinationData) |  | Проверяет наличие данных платежного средства, на которое нужно сделать выплату. |
| public | [hasPayoutToken()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_hasPayoutToken) |  | Проверяет наличие токенизированных данных для выплаты. |
| public | [hasPersonalData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_hasPersonalData) |  | Проверяет наличие персональных данных в создаваемой выплате. |
| public | [hasReceiptData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_hasReceiptData) |  | Проверяет наличие данных для формирования чека в сервисе Мой налог. |
| public | [hasSelfEmployed()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_hasSelfEmployed) |  | Проверяет наличие данных самозанятого в создаваемой выплате. |
| public | [setAmount()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_setAmount) |  | Устанавливает сумму выплаты. |
| public | [setDeal()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_setDeal) |  | Устанавливает сделку, в рамках которой нужно провести выплату. |
| public | [setDescription()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_setDescription) |  | Устанавливает описание транзакции. |
| public | [setMetadata()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_setMetadata) |  | Устанавливает метаданные, привязанные к выплате. |
| public | [setPaymentMethodId()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_setPaymentMethodId) |  | Устанавливает идентификатор сохраненного способа оплаты. |
| public | [setPayoutDestinationData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_setPayoutDestinationData) |  | Устанавливает данные платежного средства, на которое нужно сделать выплату. |
| public | [setPayoutToken()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_setPayoutToken) |  | Устанавливает токенизированные данные для выплаты. |
| public | [setPersonalData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_setPersonalData) |  | Устанавливает персональные данные получателя выплаты. |
| public | [setReceiptData()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_setReceiptData) |  | Устанавливает данные для формирования чека в сервисе Мой налог. |
| public | [setSelfEmployed()](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md#method_setSelfEmployed) |  | Устанавливает данные самозанятого, который получит выплату. |

---
### Details
* File: [lib/Request/Payouts/CreatePayoutRequestInterface.php](../../lib/Request/Payouts/CreatePayoutRequestInterface.php)
* Package: \YooKassa\Request
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |
| property |  | Сумма создаваемой выплаты |
| property |  | Данные платежного средства, на которое нужно сделать выплату. Обязательный параметр, если не передан payout_token. |
| property |  | Данные платежного средства, на которое нужно сделать выплату. Обязательный параметр, если не передан payout_token. |
| property |  | Токенизированные данные для выплаты. Например, синоним банковской карты. Обязательный параметр, если не передан payout_destination_data |
| property |  | Токенизированные данные для выплаты. Например, синоним банковской карты. Обязательный параметр, если не передан payout_destination_data |
| property |  | Идентификатор сохраненного способа оплаты, данные которого нужно использовать для проведения выплаты |
| property |  | Идентификатор сохраненного способа оплаты, данные которого нужно использовать для проведения выплаты |
| property |  | Сделка, в рамках которой нужно провести выплату. Необходимо передавать, если вы проводите Безопасную сделку |
| property |  | Описание транзакции (не более 128 символов). Например: «Выплата по договору N» |
| property |  | Данные самозанятого, который получит выплату. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат. |
| property |  | Данные самозанятого, который получит выплату. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат. |
| property |  | Данные для формирования чека в сервисе Мой налог. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат. |
| property |  | Данные для формирования чека в сервисе Мой налог. Необходимо передавать, если вы делаете выплату [самозанятому](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Только для обычных выплат. |
| property |  | Персональные данные получателя выплаты. Необходимо передавать, если вы делаете выплаты с [проверкой получателя](/developers/payouts/scenario-extensions/recipient-check) (только для выплат через СБП). |
| property |  | Персональные данные получателя выплаты. Необходимо передавать, если вы делаете выплаты с [проверкой получателя](/developers/payouts/scenario-extensions/recipient-check) (только для выплат через СБП). |
| property |  | Метаданные привязанные к выплате |

---
## Methods
<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма выплаты


<a name="method_hasAmount" class="anchor"></a>
#### public hasAmount() : bool

```php
public hasAmount() : bool
```

**Summary**

Проверяет наличие суммы в создаваемой выплате.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** bool - True если сумма установлена, false если нет


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : self

```php
public setAmount(\YooKassa\Model\AmountInterface|array $amount) : self
```

**Summary**

Устанавливает сумму выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array</code> | amount  | Сумма выплаты |

**Returns:** self - 


<a name="method_getPayoutDestinationData" class="anchor"></a>
#### public getPayoutDestinationData() : null|\YooKassa\Model\Payout\AbstractPayoutDestination

```php
public getPayoutDestinationData() : null|\YooKassa\Model\Payout\AbstractPayoutDestination
```

**Summary**

Возвращает данные платежного средства, на которое нужно сделать выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** null|\YooKassa\Model\Payout\AbstractPayoutDestination - Данные платежного средства, на которое нужно сделать выплату


<a name="method_hasPayoutDestinationData" class="anchor"></a>
#### public hasPayoutDestinationData() : bool

```php
public hasPayoutDestinationData() : bool
```

**Summary**

Проверяет наличие данных платежного средства, на которое нужно сделать выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** bool - True если данные есть, false если нет


<a name="method_setPayoutDestinationData" class="anchor"></a>
#### public setPayoutDestinationData() : self

```php
public setPayoutDestinationData(null|\YooKassa\Model\Payout\AbstractPayoutDestination|array $payout_destination_data) : self
```

**Summary**

Устанавливает данные платежного средства, на которое нужно сделать выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \YooKassa\Model\Payout\AbstractPayoutDestination OR array</code> | payout_destination_data  | Данные платежного средства, на которое нужно сделать выплату |

**Returns:** self - 


<a name="method_getPayoutToken" class="anchor"></a>
#### public getPayoutToken() : string|null

```php
public getPayoutToken() : string|null
```

**Summary**

Возвращает токенизированные данные для выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** string|null - Токенизированные данные для выплаты


<a name="method_hasPayoutToken" class="anchor"></a>
#### public hasPayoutToken() : bool

```php
public hasPayoutToken() : bool
```

**Summary**

Проверяет наличие токенизированных данных для выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** bool - True если токен установлен, false если нет


<a name="method_setPayoutToken" class="anchor"></a>
#### public setPayoutToken() : self

```php
public setPayoutToken(string|null $payout_token) : self
```

**Summary**

Устанавливает токенизированные данные для выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | payout_token  | Токенизированные данные для выплаты |

**Returns:** self - 


<a name="method_getPaymentMethodId" class="anchor"></a>
#### public getPaymentMethodId() : null|string

```php
public getPaymentMethodId() : null|string
```

**Summary**

Возвращает идентификатор сохраненного способа оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** null|string - Идентификатор сохраненного способа оплаты


<a name="method_hasPaymentMethodId" class="anchor"></a>
#### public hasPaymentMethodId() : bool

```php
public hasPaymentMethodId() : bool
```

**Summary**

Проверяет наличие идентификатора сохраненного способа оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** bool - True если идентификатора установлен, false если нет


<a name="method_setPaymentMethodId" class="anchor"></a>
#### public setPaymentMethodId() : self

```php
public setPaymentMethodId(null|string $payment_method_id) : self
```

**Summary**

Устанавливает идентификатор сохраненного способа оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | payment_method_id  | Идентификатор сохраненного способа оплаты |

**Returns:** self - 


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает описание транзакции.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** string|null - Описание транзакции


<a name="method_hasDescription" class="anchor"></a>
#### public hasDescription() : bool

```php
public hasDescription() : bool
```

**Summary**

Проверяет наличие описания транзакции в создаваемой выплате.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** bool - True если описание транзакции установлено, false если нет


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description) : self
```

**Summary**

Устанавливает описание транзакции.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Описание транзакции |

**Returns:** self - 


<a name="method_getDeal" class="anchor"></a>
#### public getDeal() : null|\YooKassa\Model\Deal\PayoutDealInfo

```php
public getDeal() : null|\YooKassa\Model\Deal\PayoutDealInfo
```

**Summary**

Возвращает сделку, в рамках которой нужно провести выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** null|\YooKassa\Model\Deal\PayoutDealInfo - Сделка, в рамках которой нужно провести выплату


<a name="method_hasDeal" class="anchor"></a>
#### public hasDeal() : bool

```php
public hasDeal() : bool
```

**Summary**

Проверяет установлена ли сделка, в рамках которой нужно провести выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** bool - True если сделка установлена, false если нет


<a name="method_setDeal" class="anchor"></a>
#### public setDeal() : self

```php
public setDeal(null|array|\YooKassa\Model\Deal\PayoutDealInfo $deal) : self
```

**Summary**

Устанавливает сделку, в рамках которой нужно провести выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Deal\PayoutDealInfo</code> | deal  | Сделка, в рамках которой нужно провести выплату |

**Returns:** self - 


<a name="method_getSelfEmployed" class="anchor"></a>
#### public getSelfEmployed() : null|\YooKassa\Request\Payouts\PayoutSelfEmployedInfo

```php
public getSelfEmployed() : null|\YooKassa\Request\Payouts\PayoutSelfEmployedInfo
```

**Summary**

Возвращает данные самозанятого, который получит выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** null|\YooKassa\Request\Payouts\PayoutSelfEmployedInfo - Данные самозанятого, который получит выплату


<a name="method_hasSelfEmployed" class="anchor"></a>
#### public hasSelfEmployed() : bool

```php
public hasSelfEmployed() : bool
```

**Summary**

Проверяет наличие данных самозанятого в создаваемой выплате.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** bool - True если данные самозанятого есть, false если нет


<a name="method_setSelfEmployed" class="anchor"></a>
#### public setSelfEmployed() : self

```php
public setSelfEmployed(null|array|\YooKassa\Request\Payouts\PayoutSelfEmployedInfo $self_employed) : self
```

**Summary**

Устанавливает данные самозанятого, который получит выплату.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Payouts\PayoutSelfEmployedInfo</code> | self_employed  | Данные самозанятого, который получит выплату |

**Returns:** self - 


<a name="method_getReceiptData" class="anchor"></a>
#### public getReceiptData() : null|\YooKassa\Request\Payouts\IncomeReceiptData

```php
public getReceiptData() : null|\YooKassa\Request\Payouts\IncomeReceiptData
```

**Summary**

Возвращает данные для формирования чека в сервисе Мой налог.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** null|\YooKassa\Request\Payouts\IncomeReceiptData - Данные для формирования чека в сервисе Мой налог


<a name="method_hasReceiptData" class="anchor"></a>
#### public hasReceiptData() : bool

```php
public hasReceiptData() : bool
```

**Summary**

Проверяет наличие данных для формирования чека в сервисе Мой налог.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** bool - True если данные для формирования чека есть, false если нет


<a name="method_setReceiptData" class="anchor"></a>
#### public setReceiptData() : self

```php
public setReceiptData(null|array|\YooKassa\Request\Payouts\IncomeReceiptData $receipt_data) : self
```

**Summary**

Устанавливает данные для формирования чека в сервисе Мой налог.

**Description**

.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Payouts\IncomeReceiptData</code> | receipt_data  | Данные для формирования чека в сервисе Мой налог |

**Returns:** self - 


<a name="method_getPersonalData" class="anchor"></a>
#### public getPersonalData() : \YooKassa\Request\Payouts\PayoutPersonalData[]|\YooKassa\Common\ListObjectInterface

```php
public getPersonalData() : \YooKassa\Request\Payouts\PayoutPersonalData[]|\YooKassa\Common\ListObjectInterface
```

**Summary**

Возвращает персональные данные получателя выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** \YooKassa\Request\Payouts\PayoutPersonalData[]|\YooKassa\Common\ListObjectInterface - Персональные данные получателя выплаты


<a name="method_hasPersonalData" class="anchor"></a>
#### public hasPersonalData() : bool

```php
public hasPersonalData() : bool
```

**Summary**

Проверяет наличие персональных данных в создаваемой выплате.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** bool - True если персональные данные есть, false если нет


<a name="method_setPersonalData" class="anchor"></a>
#### public setPersonalData() : self

```php
public setPersonalData(null|array|\YooKassa\Common\ListObjectInterface $personal_data = null) : self
```

**Summary**

Устанавливает персональные данные получателя выплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Common\ListObjectInterface</code> | personal_data  | Персональные данные получателя выплаты |

**Returns:** self - 


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает данные оплаты установленные мерчантом

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** \YooKassa\Model\Metadata|null - Метаданные привязанные к выплате


<a name="method_hasMetadata" class="anchor"></a>
#### public hasMetadata() : bool

```php
public hasMetadata() : bool
```

**Summary**

Проверяет, были ли установлены метаданные заказа.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

**Returns:** bool - True если метаданные были установлены, false если нет


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(null|array|\YooKassa\Model\Metadata $metadata) : self
```

**Summary**

Устанавливает метаданные, привязанные к выплате.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\CreatePayoutRequestInterface](../classes/YooKassa-Request-Payouts-CreatePayoutRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Metadata</code> | metadata  | Метаданные платежа, устанавливаемые мерчантом |

**Returns:** self - 




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