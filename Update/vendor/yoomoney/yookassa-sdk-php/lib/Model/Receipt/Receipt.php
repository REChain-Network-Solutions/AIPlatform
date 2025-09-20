<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Model\Receipt;

use ReturnTypeWillChange;
use YooKassa\Common\AbstractObject;
use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Validator\Constraints as Assert;
use YooKassa\Validator\Exceptions\EmptyPropertyValueException;
use YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Model\AmountInterface;

/**
 * Класс данных для формирования чека в онлайн-кассе (для соблюдения 54-ФЗ).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property ReceiptCustomer $customer Информация о плательщике
 * @property ListObjectInterface|ReceiptItemInterface[] $items Список товаров в заказе. Для чеков по 54-ФЗ: если используете Чеки от ЮKassa, можно передать максимум 80 товаров, если используете стороннюю онлайн-кассу, максимум 100 товаров.  Для чеков самозанятых — максимум 6 товаров.
 * @property ListObjectInterface|SettlementInterface[] $settlements Массив оплат, обеспечивающих выдачу товара
 * @property ListObjectInterface|ReceiptItemInterface[] $shippingItems Список товаров в заказе, являющихся доставкой
 * @property ListObjectInterface|ReceiptItemInterface[] $shipping_items Список товаров в заказе, являющихся доставкой
 * @property int $taxSystemCode Код системы налогообложения. Число 1-6.
 * @property int $tax_system_code Код системы налогообложения. Число 1-6.
 * @property AdditionalUserProps $additionalUserProps Дополнительный реквизит пользователя (тег в 54 ФЗ — 1084)
 * @property AdditionalUserProps $additional_user_props Дополнительный реквизит пользователя (тег в 54 ФЗ — 1084)
 * @property ListObjectInterface|IndustryDetails[] $receiptIndustryDetails Отраслевой реквизит чека (тег в 54 ФЗ — 1261)
 * @property ListObjectInterface|IndustryDetails[] $receipt_industry_details Отраслевой реквизит чека (тег в 54 ФЗ — 1261)
 * @property OperationalDetails $receiptOperationalDetails Операционный реквизит чека (тег в 54 ФЗ — 1270)
 * @property OperationalDetails $receipt_operational_details Операционный реквизит чека (тег в 54 ФЗ — 1270)
 */
class Receipt extends AbstractObject implements ReceiptInterface
{
    /**
     * @var ReceiptCustomer|null Информация о плательщике
     */
    #[Assert\Type(ReceiptCustomer::class)]
    #[Assert\Valid]
    private ?ReceiptCustomer $_customer = null;

    /**
     * @var ReceiptItemInterface[]|ListObjectInterface Список товаров в заказе. Для чеков по 54-ФЗ: если используете Чеки от ЮKassa, можно передать максимум 80 товаров, если используете стороннюю онлайн-кассу, максимум 100 товаров.  Для чеков самозанятых — максимум 6 товаров.
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\AllType(ReceiptItem::class)]
    #[Assert\Type(ListObject::class)]
    private ?ListObject $_items = null;

    /**
     * @var SettlementInterface[]|ListObjectInterface|null Массив оплат, обеспечивающих выдачу товара
     */
    #[Assert\Valid]
    #[Assert\AllType(Settlement::class)]
    #[Assert\Type(ListObject::class)]
    #[Assert\Count(min: 1)]
    private ?ListObject $_settlements = null;

    /**
     * @var ReceiptItemInterface[]|ListObjectInterface Список товаров в заказе, являющихся доставкой
     */
    #[Assert\Type(ListObject::class)]
    #[Assert\Valid]
    #[Assert\AllType(ReceiptItem::class)]
    private ?ListObject $_shippingItems = null;

    /**
     * @var int|null Код системы налогообложения. Число 1-6.
     */
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(1)]
    #[Assert\LessThanOrEqual(6)]
    private ?int $_tax_system_code = null;

    /** @var AdditionalUserProps|null Дополнительный реквизит пользователя (тег в 54 ФЗ — 1084) */
    #[Assert\Type(AdditionalUserProps::class)]
    #[Assert\Valid]
    private ?AdditionalUserProps $_additional_user_props = null;

    /**
     * @var IndustryDetails[]|ListObjectInterface|null Отраслевой реквизит чека (тег в 54 ФЗ — 1261)
     */
    #[Assert\Valid]
    #[Assert\AllType(IndustryDetails::class)]
    #[Assert\Type(ListObject::class)]
    private ?ListObject $_receipt_industry_details = null;

    /**
     * @var OperationalDetails|null Операционный реквизит чека (тег в 54 ФЗ — 1270)
     */
    #[Assert\Type(OperationalDetails::class)]
    #[Assert\Valid]
    private ?OperationalDetails $_receipt_operational_details = null;

    /**
     * Возвращает информацию о плательщике.
     *
     * @return ReceiptCustomer Информация о плательщике
     */
    public function getCustomer(): ReceiptCustomer
    {
        if (empty($this->_customer)) {
            $this->_customer = new ReceiptCustomer();
        }

        return $this->_customer;
    }

    /**
     * Устанавливает информацию о плательщике.
     *
     * @param ReceiptCustomer|array|null $customer
     *
     * @return self
     */
    public function setCustomer(mixed $customer = null): self
    {
        $this->_customer = $this->validatePropertyValue('_customer', $customer);
        return $this;
    }

    /**
     * Возвращает список позиций в текущем чеке.
     *
     * @return ReceiptItemInterface[]|ListObjectInterface Список товаров в заказе
     */
    public function getItems(): ListObjectInterface
    {
        if ($this->_items === null) {
            $this->_items = new ListObject(ReceiptItem::class);
        }
        return $this->_items;
    }

    /**
     * Устанавливает список позиций в чеке.
     *
     * Если до этого в чеке уже были установлены значения, они удаляются и полностью заменяются переданным списком
     * позиций. Все передаваемые значения в массиве позиций должны быть объектами класса, реализующего интерфейс
     * ReceiptItemInterface, в противном случае будет выброшено исключение InvalidPropertyValueTypeException.
     *
     * @param array|ListObjectInterface $items Список товаров в заказе: для [Чеков от ЮKassa](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/yoomoney/basics) — не более 80 товаров, для [сторонних онлайн-касс](https://yookassa.ru/developers/payment-acceptance/receipts/54fz/other-services/basics) — не более 100 товаров
     *
     * @throws EmptyPropertyValueException Выбрасывается если передали пустой массив значений
     * @throws InvalidPropertyValueTypeException Выбрасывается если в качестве значения был передан не массив и не
     *                                           итератор, либо если одно из переданных значений не реализует интерфейс ReceiptItemInterface
     *
     * @return self
     */
    public function setItems(mixed $items): self
    {
        $items = $this->validatePropertyValue('_items', $items);

        $this->_items = null;
        $this->_shippingItems = null;
        foreach ($items->getItems() as $val) {
            $this->addItem($val);
        }

        return $this;
    }

    /**
     * Обнуляет список позиций в чеке.
     *
     * Если до этого в чеке уже были установлены значения, они удаляются.
     * @return self
     */
    public function removeItems(): self
    {
        $this->getItems()->clear();

        return $this;
    }

    /**
     * Добавляет товар в чек.
     *
     * @param ReceiptItemInterface[]|ListObjectInterface $value Объект добавляемой в чек позиции
     */
    public function addItem(mixed $value): void
    {
        if ($value->isShipping()) {
            $this->getShippingItems()->add($value);
        }
        $this->getItems()->add($value);
    }

    /**
     * Возвращает массив оплат, обеспечивающих выдачу товара.
     *
     * @return SettlementInterface[]|ListObjectInterface Массив оплат, обеспечивающих выдачу товара
     */
    public function getSettlements(): ListObjectInterface
    {
        if ($this->_settlements === null) {
            $this->_settlements = new ListObject(Settlement::class);
        }
        return $this->_settlements;
    }

    /**
     * Возвращает массив оплат, обеспечивающих выдачу товара.
     *
     * @param ListObjectInterface|array|null $settlements
     *
     * @return self
     */
    public function setSettlements(mixed $settlements = null): self
    {
        $this->_settlements = $this->validatePropertyValue('_settlements', $settlements);
        return $this;
    }

    /**
     * Добавляет оплату в чек.
     *
     * @param SettlementInterface $value Оплата
     *
     * @return self
     */
    public function addSettlement(SettlementInterface $value): self
    {
        $this->getSettlements()->add($value);

        return $this;
    }

    /**
     * Возвращает список товаров в заказе, являющихся доставкой
     *
     * @return ReceiptItemInterface[]|ListObjectInterface Список товаров в заказе, являющихся доставкой
     */
    public function getShippingItems(): ListObjectInterface
    {
        if ($this->_shippingItems === null) {
            $this->_shippingItems = new ListObject(ReceiptItem::class);
        }
        return $this->_shippingItems;
    }

    /**
     * Возвращает код системы налогообложения.
     *
     * @return int|null Код системы налогообложения. Число 1-6.
     */
    public function getTaxSystemCode(): ?int
    {
        return $this->_tax_system_code;
    }

    /**
     * Устанавливает код системы налогообложения.
     *
     * @param int|null $tax_system_code Код системы налогообложения. Число 1-6
     *
     * @return self
     */
    public function setTaxSystemCode(?int $tax_system_code): self
    {
        $this->_tax_system_code = $this->validatePropertyValue('_tax_system_code', $tax_system_code);
        return $this;
    }

    /**
     * Возвращает дополнительный реквизит пользователя.
     *
     * @return AdditionalUserProps|null Дополнительный реквизит пользователя
     */
    public function getAdditionalUserProps(): ?AdditionalUserProps
    {
        return $this->_additional_user_props;
    }

    /**
     * Устанавливает дополнительный реквизит пользователя.
     *
     * @param AdditionalUserProps|array $additional_user_props Дополнительный реквизит пользователя
     *
     * @return self
     */
    public function setAdditionalUserProps(mixed $additional_user_props = null): self
    {
        $this->_additional_user_props = $this->validatePropertyValue('_additional_user_props', $additional_user_props);
        return $this;
    }

    /**
     * Возвращает отраслевой реквизит чека.
     *
     * @return IndustryDetails[]|ListObjectInterface Отраслевой реквизит чека
     */
    public function getReceiptIndustryDetails(): ListObjectInterface
    {
        if ($this->_receipt_industry_details === null) {
            $this->_receipt_industry_details = new ListObject(IndustryDetails::class);
        }
        return $this->_receipt_industry_details;
    }

    /**
     * Устанавливает отраслевой реквизит чека.
     *
     * @param array|ListObjectInterface|null $receipt_industry_details Отраслевой реквизит чека
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданный аргумент - не массив
     *
     * @return self
     */
    public function setReceiptIndustryDetails(mixed $receipt_industry_details = null): self
    {
        $this->_receipt_industry_details = $this->validatePropertyValue('_receipt_industry_details', $receipt_industry_details);
        return $this;
    }

    /**
     * Добавляет отраслевой реквизит чека.
     *
     * @param IndustryDetails|array $value Отраслевой реквизит чека.
     *
     * @return self
     */
    public function addReceiptIndustryDetails(mixed $value): self
    {
        $this->getReceiptIndustryDetails()->add($value);

        return $this;
    }

    /**
     * Возвращает операционный реквизит чека.
     *
     * @return OperationalDetails|null Операционный реквизит чека
     */
    public function getReceiptOperationalDetails(): ?OperationalDetails
    {
        return $this->_receipt_operational_details;
    }

    /**
     * Устанавливает операционный реквизит чека.
     *
     * @param array|OperationalDetails|null $receipt_operational_details Операционный реквизит чека
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается если переданный аргумент - не массив
     *
     * @return self
     */
    public function setReceiptOperationalDetails(mixed $receipt_operational_details = null): self
    {
        $this->_receipt_operational_details = $this->validatePropertyValue('_receipt_operational_details', $receipt_operational_details);
        return $this;
    }

    /**
     * Проверяет есть ли в чеке хотя бы одна позиция.
     *
     * @return bool True если чек не пуст, false если в чеке нет ни одной позиции
     */
    public function notEmpty(): bool
    {
        return !$this->getItems()->isEmpty();
    }

    /**
     * Возвращает стоимость заказа исходя из состава чека.
     *
     * @param bool $withShipping Добавить ли к стоимости заказа стоимость доставки
     *
     * @return int Общая стоимость заказа в центах/копейках
     */
    public function getAmountValue(bool $withShipping = true): int
    {
        $result = 0;
        foreach ($this->getItems() as $item) {
            if ($withShipping || !$item->isShipping()) {
                $result += $item->getAmount();
            }
        }

        return $result;
    }

    /**
     * Возвращает стоимость доставки исходя из состава чека.
     *
     * @return int Стоимость доставки из состава чека в центах/копейках
     */
    public function getShippingAmountValue(): int
    {
        $result = 0;
        foreach ($this->getItems() as $item) {
            if ($item->isShipping()) {
                $result += $item->getAmount();
            }
        }

        return $result;
    }

    /**
     * Подгоняет стоимость товаров в чеке к общей цене заказа.
     *
     * @param AmountInterface $orderAmount Общая стоимость заказа
     * @param bool $withShipping Поменять ли заодно и цену доставки
     */
    public function normalize(AmountInterface $orderAmount, bool $withShipping = false): void
    {
        $amount = $orderAmount->getIntegerValue();
        if (!$withShipping && !$this->getShippingItems()->isEmpty()) {
            if ($amount > $this->getShippingAmountValue()) {
                $amount -= $this->getShippingAmountValue();
            } else {
                $withShipping = true;
            }
        }
        $realAmount = $this->getAmountValue($withShipping);

        if ($realAmount !== $amount) {
            $coefficient = (float) $amount / (float) $realAmount;
            $items = [];
            $realAmount = 0;
            foreach ($this->getItems() as $item) {
                if ($withShipping || !$item->isShipping()) {
                    $price = round($coefficient * $item->getPrice()->getIntegerValue());
                    if ($price < 1.0) {
                        if ($item->getPrice()->getIntegerValue() > 1) {
                            $item->getPrice()->setValue(0.01);
                        }
                        $amount -= $item->getAmount();
                    } else {
                        $items[] = $item;
                        $realAmount += $item->getAmount();
                    }
                }
            }
            uasort($items, static function (ReceiptItemInterface $a, ReceiptItemInterface $b) {
                if ($a->getPrice()->getIntegerValue() > $b->getPrice()->getIntegerValue()) {
                    return -1;
                }
                if ($a->getPrice()->getIntegerValue() < $b->getPrice()->getIntegerValue()) {
                    return 1;
                }

                return 0;
            });

            $coefficient = (float) $amount / (float) $realAmount;
            $realAmount = 0;
            $aloneId = null;
            foreach ($items as $index => $item) {
                if ($withShipping || !$item->isShipping()) {
                    $item->applyDiscountCoefficient($coefficient);
                    $realAmount += $item->getAmount();
                    if (null === $aloneId && 1.0 === $item->getQuantity() && !$item->isShipping()) {
                        $aloneId = $index;
                    }
                }
            }
            if (null === $aloneId) {
                foreach ($this->getItems() as $index => $item) {
                    if (!$item->isShipping()) {
                        $aloneId = $index;

                        break;
                    }
                }
            }
            if (null === $aloneId) {
                $aloneId = 0;
            }
            $diff = $amount - $realAmount;
            if (abs($diff) >= 0.1) {
                if (1.0 === $this->_items[$aloneId]->getQuantity()) {
                    $this->_items[$aloneId]->increasePrice($diff / 100.0);
                } elseif ($this->_items[$aloneId]->getQuantity() > 1.0) {
                    $item = $this->_items[$aloneId]->fetchItem(1);
                    $item->increasePrice($diff / 100.0);
                    $items = $this->getItems()->toArray();
                    array_splice($items, $aloneId + 1, 0, [$item]);
                    $this->_items = new ListObject(ReceiptItem::class, $items);
                } else {
                    $item = $this->_items[$aloneId]->fetchItem($this->_items[$aloneId]->getQuantity() / 2);
                    $item->increasePrice($diff / 100.0);
                    $items = $this->getItems()->toArray();
                    array_splice($items, $aloneId + 1, 0, [$item]);
                    $this->_items = new ListObject(ReceiptItem::class, $items);
                }
            }
        }
    }

    /**
     * Возвращает Id объекта чека.
     *
     * @return null|string Id объекта чека
     */
    public function getObjectId(): ?string
    {
        return null;
    }

    #[ReturnTypeWillChange]
    /**
     * Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.
     *
     * @return array Ассоциативный массив со свойствами текущего объекта
     */
    public function jsonSerialize(): array
    {
        $data = parent::jsonSerialize();
        unset($data['amount_value'], $data['shipping_amount_value']);

        return $data;
    }
}
