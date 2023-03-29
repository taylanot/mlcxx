
alias pwd='pwd -P'

quote()
{
  echo "'$(echo "$1" | sed -e "s/'/'\\\\''/g")'"
}
