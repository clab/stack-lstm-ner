#!/usr/bin/perl -w
use strict;

my @toks;
my @tags;

while(<>) {
  chomp;
  if (/^\s*$/) {
    print "@toks ||| ";
    my $len = scalar @toks;
    my $i = 0;
    while ($i < $len) {
      if ($tags[$i] eq 'O') {
        print "OUT ";
        $i++;
      } elsif ($tags[$i] =~ /^(B|I)-(.+)$/) {
        my $tt = $2;
        my $x = "I-$tt";
        my $j = $i + 1;
        while ($j < $len && $tags[$j] eq $x) { $j++; }
        my @span = ();
        for (my $k = $i; $k < $j; $k++) {
          print "SHIFT ";
        }
        print "REDUCE($tt) ";
        $i = $j;
      } else {
        die "Bad input: $_\n";
      }
    }
    @toks = ();
    @tags = ();
    print "\n";
  } else {
    my @fields = split /\s+/;
    push @toks, "$fields[0]-$fields[1]";
    push @tags, $fields[-1];
  }
}

